"""
Router — Language detection and legal relevance classification.

Acts as the gatekeeper: determines the user's language and whether the query
is legal in nature before allowing further processing.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from utils.config import get_llm
from controllers.base_agent import BaseAgent


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal query router. Your job is to:
1. Detect the language of the user's query (Arabic or English).
2. Determine if the query is related to law, legal matters, or the Egyptian Civil Code.
3. Assess your confidence in the classification (high, medium, or low).

Respond in EXACTLY this JSON format (no markdown, no code blocks):
{{"language": "ar" or "en", "is_legal": true or false, "reason": "brief explanation", "confidence": 0.0 to 1.0}}

confidence guidelines:
- high (0.8-1.0): Clear legal topic or clearly non-legal (greetings, weather, etc.)
- medium (0.5-0.79): Ambiguous topics that might be related to law
- low (0.0-0.49): Very unclear intent, requires more research

Examples of LEGAL queries (high confidence):
- "ما هي شروط صحة العقد؟" → {{"language": "ar", "is_legal": true, "reason": "Asks about contract validity conditions", "confidence": 0.95}}
- "What are property rights?" → {{"language": "en", "is_legal": true, "reason": "Asks about legal property rights", "confidence": 0.9}}

Examples of Neutral queries:
- "Hello"→ {{"language": "en", "is_legal": false, "reason": "Simple greeting", "confidence": 0.95}}
- "مرحبا"→ {{"language": "ar", "is_legal": false, "reason": "Simple greeting", "confidence": 0.95}}

Examples of NON-LEGAL queries:
- "What's the weather today?" → {{"language": "en", "is_legal": false, "reason": "Weather is not legal topic", "confidence": 0.95}}
- "اكتب لي قصيدة" → {{"language": "ar", "is_legal": false, "reason": "Poetry request is not legal", "confidence": 0.9}}
- "How do I cook pasta?" → {{"language": "en", "is_legal": false, "reason": "Cooking is not legal", "confidence": 0.95}}

Examples of Medium confidence:
- "ما هو الحكم؟" → {{"language": "ar", "is_legal": true, "reason": "Could be legal ruling or court judgment, but unclear", "confidence": 0.6}}
- "Tell me about the law" → {{"language": "en", "is_legal": true, "reason": "Vague but likely legal intent", "confidence": 0.65}}
"""),
    ("human", "{query}"),
])


class RouterAgent(BaseAgent):
    """
    Agent responsible for classifying query's language and determining
    if the query is legal in nature.
    """
    
    def execute(self, query: str) -> dict:
        """
        Classify a user query for language and legal relevance.
        
        Args:
            query: The user's input text.
        
        Returns:
            Dict with keys: language ("ar"/"en"), is_legal (bool), reason (str)
        """
        llm = get_llm()
        chain = ROUTER_PROMPT | llm

        response = chain.invoke({"query": query})
        response_text = response.content.strip()

        # Parse the JSON response
        try:
            # Clean potential markdown code blocks
            cleaned = response_text
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            return {
                "language": result.get("language", "ar"),
                "is_legal": result.get("is_legal", True),
                "reason": result.get("reason", ""),
                "confidence": float(result.get("confidence", 0.5)),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: assume legal and medium confidence if parsing fails
            return {
                "language": "ar",
                "is_legal": True,
                "reason": "Router parse fallback — treating as legal query.",
                "confidence": 0.5,
            }
