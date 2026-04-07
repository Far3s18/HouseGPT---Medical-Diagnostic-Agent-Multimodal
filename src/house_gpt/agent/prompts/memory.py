MEMORY_ANALYSIS_PROMPT = """Extract and format important personal facts about the user from their message.
Focus on the actual information, not meta-commentary or requests.

Important facts include:
- Personal details (name, age, location)
- Professional info (job, education, skills)
- Preferences (likes, dislikes, favorites)
- Life circumstances (family, relationships)
- Significant experiences or achievements
- Personal goals or aspirations

Rules:
1. Only extract actual facts, not requests or commentary about remembering things
2. Convert facts into clear, third-person statements
3. If no actual facts are present, mark as not important
4. Remove conversational elements and focus on the core information

Examples:
Input: "Hey, could you remember that I love Star Wars?"
Output: {{
    "is_important": true,
    "formatted_memory": "Loves Star Wars"
}}

Input: "Please make a note that I work as an engineer"
Output: {{
    "is_important": true,
    "formatted_memory": "Works as an engineer"
}}

Input: "Remember this: I live in Madrid"
Output: {{
    "is_important": true,
    "formatted_memory": "Lives in Madrid"
}}

Input: "Can you remember my details for next time?"
Output: {{
    "is_important": false,
    "formatted_memory": null
}}

Input: "Hey, how are you today?"
Output: {{
    "is_important": false,
    "formatted_memory": null
}}

Input: "I studied computer science at MIT and I'd love if you could remember that"
Output: {{
    "is_important": true,
    "formatted_memory": "Studied computer science at MIT"
}}

Message: {message}
Output:
"""