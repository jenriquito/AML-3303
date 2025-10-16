"""
Lambton College Ottawa - Campus Survival Guide Chatbot
Streamlit Web Application - Software Tools and Emerging Technologies for AI and ML (AML-3303)
Course Instructor: Akshey Singhal
Student: Enrique Fernandez C.
Date: October 2025

"""

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ============================================================================
# STEP 1: PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Lambton Ottawa Guide",
    page_icon="🎓",
    layout="wide"
)

# ============================================================================
# STEP 2: OUR FAQ DATASET
# ============================================================================

faq_text = """
Q: Where is Lambton College Ottawa located?
Q: ¿Dónde está ubicado Lambton College Ottawa?
A: Lambton College Ottawa is located at 223 Main Street, Ottawa, ON K1S 1C4, on the Saint Paul University campus in the heart of Canada's capital.

Q: How much does student housing cost in Ottawa?
Q: ¿Cuánto cuesta el alojamiento estudiantil en Ottawa?
A: On-campus residence typically costs between $800-$1200 per month including utilities. Off-campus shared apartments range from $600-$900 per month per room.

Q: How does public transportation work in Ottawa for students?
Q: ¿Cómo funciona el transporte público en Ottawa para estudiantes?
A: Ottawa uses OC Transpo buses and O-Train light rail. Students can get a U-Pass for approximately $229 per term. You'll need a Presto card which costs $4.

Q: Where are the cheapest grocery stores for students in Ottawa?
Q: ¿Dónde están los supermercados más baratos para estudiantes en Ottawa?
A: The most affordable grocery stores are No Frills, Food Basics (10% student discount on select days), Walmart, and FreshCo. Avoid Metro and Loblaws as they're more expensive.

Q: Can I work while studying at Lambton College Ottawa?
Q: ¿Puedo trabajar mientras estudio en Lambton College Ottawa?
A: Yes! International students can work off-campus up to 24 hours per week during academic sessions. You can work full-time during scheduled breaks.

Q: What is UHIP and do I need it as an international student?
Q: ¿Qué es UHIP y lo necesito como estudiante internacional?
A: UHIP is mandatory health insurance for international students in Ontario. It covers doctor visits, emergency care, and hospitalization. Your college automatically enrolls you.
"""

# ============================================================================
# STEP 3: SPLIT TEXT INTO LINES
# ============================================================================

lines = [line.strip() for line in faq_text.split("\n") if line.strip()]

# ============================================================================
# STEP 4: LOAD MODEL AND CREATE INDEX
# ============================================================================

@st.cache_resource
def load_everything():
    """
    This function loads the model and creates the FAISS index
    It only runs ONCE (cached) to make the app faster
    """
    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for all lines
    embeddings = model.encode(lines)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return model, index

# Load model and index
model, index = load_everything()

# ============================================================================
# STEP 5: CREATE THE WEB PAGE
# ============================================================================

# Main title
st.title("Lambton College Ottawa - Campus Survival Guide")

# Subtitle
st.markdown("### Bilingual Chatbot for International Students")

st.markdown("")

# Information box
st.info("💡 **Tip:** You can ask questions in English or Spanish.")

st.markdown("---")

# ============================================================================
# STEP 6: LANGUAGE SELECTOR
# ============================================================================

# Columns for language selection
col1, col2 = st.columns(2)

with col1:
    if st.button("🇬🇧 English", use_container_width=True):
        st.session_state.language = "English"

with col2:
    if st.button("🇪🇸 Español", use_container_width=True):
        st.session_state.language = "Spanish"

# Set default language if not selected
if 'language' not in st.session_state:
    st.session_state.language = "English"

# Show current language
if st.session_state.language == "English":
    st.success("✅ Current language: English")
else:
    st.success("✅ Idioma actual: Español")

st.markdown("---")

# ============================================================================
# STEP 7: SUGGESTED QUESTIONS
# ============================================================================

st.subheader("Quick Questions - Click to try:")

# Different questions based on language
if st.session_state.language == "English":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📍 - Where is campus?"):
            st.session_state.question = "Where is the campus located?"
        if st.button("🏠 - Housing cost?"):
            st.session_state.question = "How much is housing?"
    
    with col2:
        if st.button("🚌 - Public transit?"):
            st.session_state.question = "How does public transit work?"
        if st.button("🛒 - Cheap groceries?"):
            st.session_state.question = "Where to buy cheap groceries?"
    
    with col3:
        if st.button("💼 - Can I work?"):
            st.session_state.question = "Can I work while studying?"
        if st.button("🏥 - What is UHIP?"):
            st.session_state.question = "What is UHIP?"

else:  # Spanish
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📍 - ¿Dónde está el campus?"):
            st.session_state.question = "¿Dónde está ubicado el campus?"
        if st.button("🏠 - ¿Costo de alojamiento?"):
            st.session_state.question = "¿Cuánto cuesta el alojamiento?"
    
    with col2:
        if st.button("🚌 - ¿Transporte público?"):
            st.session_state.question = "¿Cómo funciona el transporte público?"
        if st.button("🛒 - ¿Supermercados baratos?"):
            st.session_state.question = "¿Dónde comprar comida barata?"
    
    with col3:
        if st.button("💼 - ¿Puedo trabajar?"):
            st.session_state.question = "¿Puedo trabajar mientras estudio?"
        if st.button("🏥 - ¿Qué es UHIP?"):
            st.session_state.question = "¿Qué es UHIP?"

st.markdown("---")

# ============================================================================
# STEP 8: QUESTION INPUT BOX
# ============================================================================

# Get the question
if 'question' not in st.session_state:
    st.session_state.question = ""

# Text input for custom questions
if st.session_state.language == "English":
    user_question = st.text_input(
        "Or type your own question:",
        value=st.session_state.question,
        placeholder="Example: Where can I find affordable food?"
    )
else:
    user_question = st.text_input(
        "O escribe tu propia pregunta:",
        value=st.session_state.question,
        placeholder="Ejemplo: ¿Dónde puedo encontrar comida económica?"
    )

# Search button
if st.session_state.language == "English":
    search_button = st.button("🔍 Search Answer", type="primary", use_container_width=True)
else:
    search_button = st.button("🔍 Buscar Respuesta", type="primary", use_container_width=True)

# ============================================================================
# STEP 9: SEARCH AND DISPLAY ANSWER
# ============================================================================

if search_button and user_question:
    
    # Show a loading message
    with st.spinner("🔍 Searching..."):
        
        # Convert user question to vector
        q_emb = model.encode([user_question])
        
        # Search in FAISS index for 5 most similar
        D, I = index.search(np.array(q_emb), k=5)
        
        # Set confidence threshold
        threshold = 1.5
        
        # Look for an answer line (A:)
        answer_found = None
        distance = 999
        
        for idx, dist in zip(I[0], D[0]):
            line = lines[idx]
            
            # Check if distance is too high (not relevant)
            if dist > threshold:
                if st.session_state.language == "English":
                    answer_found = "I don't have information about that topic in my database. Please ask about: housing, transportation, groceries, work permits, or UHIP."
                else:
                    answer_found = "No tengo información sobre ese tema en mi base de datos. Por favor pregunta sobre: alojamiento, transporte, supermercados, permisos de trabajo, o UHIP."
                distance = dist
                break
            
            # If we found an answer, save it
            if line.startswith("A:"):
                answer_found = line
                distance = dist
                break
            
            # If we found a question, check the next lines for answer
            if line.startswith("Q:"):
                # Check if next line is answer
                if idx + 1 < len(lines) and lines[idx + 1].startswith("A:"):
                    answer_found = lines[idx + 1]
                    distance = dist
                    break
                # Sometimes there are 2 questions (bilingual), check 2 lines ahead
                if idx + 2 < len(lines) and lines[idx + 2].startswith("A:"):
                    answer_found = lines[idx + 2]
                    distance = dist
                    break
    
    # Display the result
    st.markdown("---")
    
    if answer_found:
        # Remove "A:" from the answer if it exists
        if answer_found.startswith("A:"):
            clean_answer = answer_found.replace("A:", "").strip()
        else:
            clean_answer = answer_found  # Already clean (threshold message)
        
        # Show confidence level based on distance
        if distance > threshold:
            # Question not in database
            confidence = "🔴 Topic Not Found"
            box_type = "error"
        elif distance < 1.0:
            confidence = "🟢 High Confidence"
            box_type = "success"
        elif distance < 1.5:
            confidence = "🟡 Medium Confidence"
            box_type = "info"
        else:
            confidence = "🔴 Low Confidence"
            box_type = "warning"
        
        # Display result
        st.subheader("Answer:")
        
        if box_type == "success":
            st.success(clean_answer)
        elif box_type == "info":
            st.info(clean_answer)
        elif box_type == "error":
            st.error(clean_answer)
        else:
            st.warning(clean_answer)
        
        # Show confidence
        st.caption(f"{confidence} (Distance: {distance:.2f})")
        
    else:
        if st.session_state.language == "English":
            st.error("Sorry, I couldn't find a relevant answer. Try asking differently!")
        else:
            st.error("Lo siento, no encontré una respuesta relevante. ¡Intenta preguntar de otra forma!")

elif search_button and not user_question:
    if st.session_state.language == "English":
        st.warning("Please enter a question first!")
    else:
        st.warning("¡Por favor ingresa una pregunta primero!")

# ============================================================================
# STEP 10: FOOTER WITH INFO
# ============================================================================

st.markdown("---")
st.markdown("### Chatbot DEtails")

# Columns for stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("FAQ Topics", "6")

with col2:
    st.metric("Languages", "2")

with col3:
    st.metric("Vector Size", "384")
