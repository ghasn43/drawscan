"""
comparator_app.py - Standalone BOQ Comparison App
"""
import streamlit as st
from boq_comparator import StreamlitBOQComparator

# Page config
st.set_page_config(
    page_title="BOQ Comparator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3498db;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">⚖️ DXF BOQ Comparator</h1>', unsafe_allow_html=True)
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📖 How to Use")
        
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Upload** both BOQ Excel files:
           - 📋 Client's original BOQ
           - 👷 Contractor's calculated BOQ
        
        2. **Configure** comparison settings:
           - Set tolerance percentage
           - Select sheets if needed
        
        3. **Click** "Compare BOQs" button
        
        4. **Review** results:
           - Summary metrics
           - Visual charts
           - Detailed comparison table
        
        5. **Export** reports for meetings
        
        ### Use Cases:
        - 🏗️ Contractor bid validation
        - 📊 Client audit checks
        - 🔍 Quantity surveying
        - 💰 Payment certification
        - ⚖️ Dispute resolution
        """)
        
        st.divider()
        
        st.header("📁 Sample Files")
        
        col_sample1, col_sample2 = st.columns(2)
        with col_sample1:
            if st.button("Get Sample Client BOQ", use_container_width=True):
                st.info("Sample files feature coming soon!")
        
        with col_sample2:
            if st.button("Get Sample Contractor BOQ", use_container_width=True):
                st.info("Sample files feature coming soon!")
    
    # Main content
    comparator = StreamlitBOQComparator()
    comparator.render_ui()
    
    # Footer
    st.divider()
    col_foot1, col_foot2, col_foot3 = st.columns(3)
    
    with col_foot1:
        st.markdown("**Accuracy**: ±0.5% tolerance")
    
    with col_foot2:
        st.markdown("**Formats**: Excel (.xlsx, .xls)")
    
    with col_foot3:
        st.markdown("**Version**: 1.0.0")

if __name__ == "__main__":
    main()