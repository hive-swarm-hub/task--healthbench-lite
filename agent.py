"""HealthBench Lite — Health Assistant Agent

This agent receives a multi-turn health conversation and generates a response.
The response is graded against physician-written rubrics covering accuracy,
completeness, safety, communication, and instruction following.

Modify this file to improve the score. Everything is fair game: prompts,
pipeline stages, model selection, domain detection, response structure.
"""

import json
import os
import re
import sys

from openai import OpenAI

# ── CONFIG ───────────────────────────────────────────────────────────────────

MODEL = os.environ.get("SOLVER_MODEL", "gpt-4.1-mini")
CRITIQUE_MODEL = os.environ.get("CRITIQUE_MODEL", MODEL)
TEMPERATURE = 0.3
MAX_TOKENS = 2048
ENABLE_SELF_REFINE = True  # toggle self-critique pipeline
ENABLE_DOMAIN_DETECTION = True  # toggle domain-specific prompting

# ── DOMAIN DETECTION ─────────────────────────────────────────────────────────

DOMAINS = {
    "emergency": {
        "keywords": ["emergency", "chest pain", "stroke", "bleeding", "unconscious",
                     "seizure", "anaphylaxis", "heart attack", "can't breathe",
                     "severe pain", "overdose", "poisoning", "suicidal"],
        "instructions": (
            "This may be a medical emergency. Lead with urgency.\n"
            "- If life-threatening: IMMEDIATELY advise calling emergency services (911/112)\n"
            "- Provide first-aid steps while waiting for help\n"
            "- Be direct and action-oriented, not conversational\n"
            "- Do NOT downplay symptoms that could be serious"
        ),
    },
    "mental_health": {
        "keywords": ["depression", "anxiety", "suicid", "self-harm", "panic attack",
                     "eating disorder", "ptsd", "trauma", "bipolar", "schizophren",
                     "lonely", "hopeless", "worthless", "mental health"],
        "instructions": (
            "This involves mental health. Prioritize empathy and safety.\n"
            "- Validate their feelings before offering information\n"
            "- If there's any risk of self-harm: provide crisis resources (988 Lifeline)\n"
            "- Avoid minimizing ('just cheer up') or being overly clinical\n"
            "- Suggest professional mental health support gently\n"
            "- Use warm, non-judgmental language"
        ),
    },
    "medication": {
        "keywords": ["medication", "drug", "dosage", "side effect", "prescription",
                     "interaction", "pill", "tablet", "antibiotic", "ibuprofen",
                     "acetaminophen", "contraindication", "allergic reaction"],
        "instructions": (
            "This involves medication. Be precise and safety-conscious.\n"
            "- Always recommend consulting their prescribing physician\n"
            "- Mention common side effects AND serious warning signs\n"
            "- Flag potential drug interactions if multiple medications mentioned\n"
            "- Never recommend changing dosage without medical supervision\n"
            "- Include when to seek immediate medical attention"
        ),
    },
    "chronic": {
        "keywords": ["diabetes", "hypertension", "asthma", "arthritis", "copd",
                     "chronic", "long-term", "management", "blood sugar", "blood pressure",
                     "cholesterol", "thyroid", "autoimmune"],
        "instructions": (
            "This involves chronic condition management.\n"
            "- Provide practical, actionable lifestyle guidance\n"
            "- Emphasize regular monitoring and medical follow-up\n"
            "- Discuss both medication and non-medication approaches\n"
            "- Be encouraging about manageable changes\n"
            "- Mention when to escalate to their specialist"
        ),
    },
    "pediatric": {
        "keywords": ["child", "baby", "infant", "toddler", "pediatric", "my son",
                     "my daughter", "kid", "newborn", "teenager", "adolescent",
                     "vaccination", "fever in child"],
        "instructions": (
            "This involves a child/pediatric case.\n"
            "- Dosages and treatments differ significantly for children\n"
            "- Always recommend consulting a pediatrician\n"
            "- Provide age-appropriate guidance\n"
            "- Address parent anxiety with reassurance where appropriate\n"
            "- Flag symptoms that require urgent pediatric evaluation"
        ),
    },
}


def detect_domain(messages: list[dict]) -> str:
    """Detect the health domain from conversation content."""
    text = " ".join(m.get("content", "") for m in messages).lower()
    scores = {}
    for domain, info in DOMAINS.items():
        scores[domain] = sum(1 for kw in info["keywords"] if kw in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── PROMPTS ──────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are an expert health assistant providing accurate, thorough, and empathetic medical guidance.

## Response Principles
1. **Accuracy first**: Only provide medically accurate information. If uncertain, say so.
2. **Be thorough**: Cover all relevant aspects — causes, symptoms, treatments, when to seek help.
3. **Safety conscious**: Include appropriate warnings, contraindications, and red flags.
4. **Empathetic tone**: Acknowledge the person's concern before diving into information.
5. **Actionable**: Give clear, specific next steps they can take.
6. **Appropriate disclaimers**: Recommend professional consultation without being dismissive.

## Response Structure
When appropriate, organize your response as:
- **Acknowledgment**: Brief empathetic response to their concern
- **Key Information**: Direct answer to their question with medical accuracy
- **Important Considerations**: Side effects, risks, contraindications, or nuances
- **When to Seek Help**: Specific warning signs that need professional attention
- **Next Steps**: Clear, actionable recommendations

## Safety Rules
- For potential emergencies: lead with "Seek immediate medical attention" or "Call emergency services"
- Never diagnose definitively — frame as possibilities that need professional evaluation
- Never recommend stopping prescribed medication without medical supervision
- Always mention relevant drug interactions when discussing medications
- For mental health crises: provide crisis resources (988 Suicide & Crisis Lifeline)"""

DOMAIN_TEMPLATE = """{base_prompt}

## Domain-Specific Guidance
{domain_instructions}"""

CRITIQUE_PROMPT = """You are a physician reviewing an AI health assistant's response.

## Conversation
{conversation}

## AI Response to Review
{response}

## Review Criteria
Evaluate the response on:
1. **Medical accuracy** — Is the information correct? Any errors or misleading claims?
2. **Completeness** — Does it address all aspects of the question? Missing anything important?
3. **Safety** — Are appropriate warnings included? Any dangerous omissions?
4. **Communication** — Is the tone empathetic? Is it clear and well-organized?
5. **Appropriateness** — Does it recommend professional help when needed? Proper disclaimers?

## Task
List specific improvements needed. Be concrete — say exactly what should be added, removed, or changed.
If the response is good, say "No changes needed."
Keep your review under 200 words."""

REFINE_PROMPT = """You are an expert health assistant. You previously gave a response that received feedback from a physician reviewer.

## Original Conversation
{conversation}

## Your Previous Response
{response}

## Physician Feedback
{critique}

## Task
Rewrite your response incorporating the physician's feedback. Maintain the same helpful tone but fix all identified issues. If the feedback says "No changes needed," return the original response unchanged."""

# ── PIPELINE ─────────────────────────────────────────────────────────────────


def build_system_prompt(messages: list[dict]) -> str:
    """Build a system prompt, optionally with domain-specific instructions."""
    if not ENABLE_DOMAIN_DETECTION:
        return BASE_SYSTEM_PROMPT

    domain = detect_domain(messages)
    if domain == "general" or domain not in DOMAINS:
        return BASE_SYSTEM_PROMPT

    return DOMAIN_TEMPLATE.format(
        base_prompt=BASE_SYSTEM_PROMPT,
        domain_instructions=DOMAINS[domain]["instructions"],
    )


def format_conversation(messages: list[dict]) -> str:
    """Format messages for the critique prompt."""
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def generate(client: OpenAI, messages: list[dict], system_prompt: str) -> str:
    """Generate initial response."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model=MODEL,
        messages=full_messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


def critique(client: OpenAI, conversation_str: str, response: str) -> str:
    """Get physician-style critique of the response."""
    prompt = CRITIQUE_PROMPT.format(conversation=conversation_str, response=response)

    result = client.chat.completions.create(
        model=CRITIQUE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return result.choices[0].message.content


def refine(client: OpenAI, messages: list[dict], response: str, critique_text: str) -> str:
    """Refine response based on critique."""
    if "no changes needed" in critique_text.lower():
        return response

    conversation_str = format_conversation(messages)
    prompt = REFINE_PROMPT.format(
        conversation=conversation_str,
        response=response,
        critique=critique_text,
    )

    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return result.choices[0].message.content


def generate_response(messages: list[dict]) -> str:
    """Full pipeline: generate → critique → refine."""
    client = OpenAI()
    system_prompt = build_system_prompt(messages)

    # Step 1: Generate initial response
    response = generate(client, messages, system_prompt)

    # Step 2: Self-refine (critique + improve)
    if ENABLE_SELF_REFINE:
        conversation_str = format_conversation(messages)
        critique_text = critique(client, conversation_str, response)
        response = refine(client, messages, response, critique_text)

    return response


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    problem = json.loads(sys.stdin.read())
    messages = problem["prompt"]
    response = generate_response(messages)
    print(response)
