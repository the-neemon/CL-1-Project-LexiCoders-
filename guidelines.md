# Annotation Guidelines for Named Entity Recognition in Hindi-English Code-mixed Text

## 1. Introduction

These guidelines ensure consistency in annotating named entities in Hindi-English code-mixed text, addressing the specific challenges of mixed language text.

## 2. Entity Types

- **Person (PER)**: Names of individuals
- **Location (LOC)**: Geographical locations, buildings, landmarks
- **Organization (ORG)**: Companies, institutions, agencies
- **Date/Time (DATE)**: Temporal expressions
- **Culturally-Specific Entities (CSE)**: Cultural events, festivals, rituals
- **Products/Brands (PROD)**: Commercial products, brands, services

## 3. Entity Boundaries in Mixed-Language Contexts

### 3.1 Basic Principles
- Annotate the entire span of a named entity as a single entity, regardless of language shifts
- Include articles and function words only if they're part of the official name

### 3.2 Mixed-Language Named Entities
- **Example**: "State Bank of India ke branch" → Only "State Bank of India" is ORG
- **Example**: "delhi university mei" → Only "delhi university" is ORG

### 3.3 Nested Entities
- **Example**: "University of Delhi ke History Department" → "University of Delhi" as ORG and "History Department" as ORG

## 4. Handling Transliteration Variations

### 4.1 Standardization Principle
- Same entity should be annotated consistently regardless of transliteration variations

### 4.2 Common Variations
- "Shahrukh"/"Shah Rukh"/"Sharukh" → All variants are PER
- "Dilli"/"Delhi" → Both are LOC

### 4.3 Phonetic Variations
- v/w substitutions: "Varanasi"/"Waranasi"
- sh/s variations: "Ashok"/"Asok"
- a/u variations: "Calcutta"/"Kolkata"

### 4.4 Script Mixing Within Entities
- **Example**: "राम college" → Entire phrase is ORG if referring to "Ram College"

## 5. Culturally-Specific Entities Classification

### 5.1 Festival and Cultural Events (CSE)
- "Diwali"/"दिवाली", "Eid"/"ईद", "Durga Puja"/"दुर्गा पूजा" → CSE

### 5.2 Religious and Spiritual References
- "Shiva"/"शिव", "Krishna"/"कृष्ण" → CSE (Exception: When used for humans, annotate as PER)

### 5.3 Traditional Practices and Rituals
- "Kathak"/"कथक" (dance form), "Mehndi"/"मेहंदी" (ritual) → CSE

### 5.4 Regional Cuisine and Food Items
- "Biryani"/"बिरयानी", "Chaat"/"चाट" → CSE (Exception: "Delhi Biryani House" would be ORG)

## 6. Script Variation Handling

### 6.1 Consistent Entity Recognition Across Scripts
- Recognize entities regardless of script: Roman, Devanagari, or mixed

### 6.2 Script-Specific Guidelines
- **Devanagari**: "भारतीय जनता पार्टी" → ORG
- **Roman**: "delhi" vs. "Delhi" → Both are LOC regardless of capitalization
- **Mixed-Script**: "राम Stadium" → Annotate entire phrase if it's a single entity

### 6.3 Abbreviated and Acronym Forms
- "BJP"/"बीजेपी", "UPA"/"यूपीए" → ORG

## 7. Special Cases and Ambiguities

### 7.1 Ambiguous Entities
- Prioritize the most specific category
- Use context to determine the intended reference

### 7.2 Metonymic Uses
- **Example**: "White House ne kaha" → "White House" is ORG (not LOC)

### 7.3 Social Media Handles and Usernames
- "@narendramodi" → PER
- "@BJP4India" → ORG

## 8. Annotation Process Guidelines

1. Read entire text for context
2. Identify potential entities
3. Resolve ambiguities
4. Mark boundaries precisely
5. Assign appropriate entity type
6. Document uncertain cases
7. Ensure consistency across instances

## 9. Example Annotations

- "main kal mumbai airport gaya tha" → "mumbai airport" is LOC
- "dilli university mein admission liya" → "dilli university" is ORG
- "मैंने अपना नया iPhone खरीदा" → "iPhone" is PROD
- "kal holi celebrate karenge" → "holi" is CSE
- "State Bank of India ke Delhi branch" → "State Bank of India" is ORG, "Delhi" is LOC

/home/yash-more/Downloads/iss/labs/lab_26_march/guidelines.md