# """
# streamlit_app.py - DXF BOQ Extractor Pro with BOQ Comparator
# """
# import streamlit as st
# import pandas as pd
# import json
# from io import BytesIO
# import sys
# import os
# import tempfile

# # Add the current directory to the path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Import all modules
# from dxf_processor import DXFProcessor
# from boq_generator import BOQGenerator
# from visualization import DXFVisualizer, get_matplotlib_fig_bytes
# from boq_comparator import StreamlitBOQComparator
import streamlit as st
import pandas as pd
import json
from io import BytesIO
import sys
import os
import tempfile

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all modules
from dxf_processor import DXFProcessor
from boq_generator import BOQGenerator
from visualization import DXFVisualizer, get_matplotlib_fig_bytes
from boq_comparator import StreamlitBOQComparator

# Page config
st.set_page_config(
    page_title="DXF BOQ Extractor Pro",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/dxf-boq-extractor',
        'Report a bug': 'https://github.com/yourusername/dxf-boq-extractor/issues',
        'About': """
        # DXF BOQ Extractor Pro v2.0
        
        Professional tool for extracting Bill of Quantities from DXF drawings
        and comparing BOQs between clients and contractors.
        
        Features:
        - 🏗️ Automatic wall extraction from DXF files
        - 📊 Accurate quantity calculations
        - 💰 Cost estimation with customizable rates
        - ⚖️ BOQ comparison and discrepancy detection
        - 📈 Interactive 2D/3D visualizations
        - 📤 Multiple export formats
        
        Developed for construction professionals.
        """
    }
)

# Custom CSS
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sub headers */
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #3498db;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #3498db;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        color: #2c3e50;
        font-weight: 700;
    }
    
    .metric-unit {
        font-size: 0.9rem;
        color: #95a5a6;
        margin-left: 0.3rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 2px dashed #3498db;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #3498db;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #2c3e50;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
visualizer = DXFVisualizer()
comparator = StreamlitBOQComparator()

def render_dxf_processor():
    """Render DXF Processing Tab"""
    st.markdown('<h2 class="sub-header">🏗️ DXF to BOQ Processor</h2>', unsafe_allow_html=True)
    
    col_main, col_side = st.columns([3, 1])
    
    with col_side:
        st.markdown("### ⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "**Upload DXF File**", 
            type=['dxf'],
            help="Upload your architectural DXF drawing file"
        )
        
        if uploaded_file:
            st.success(f"**✅ File Ready:** {uploaded_file.name}")
            file_size = len(uploaded_file.getbuffer()) / 1024
            st.caption(f"File size: {file_size:.1f} KB")
        
        # Material settings
        with st.expander("💰 **Material Rates**", expanded=True):
            brick_rate = st.number_input(
                "Brickwork Rate (₹/m³)", 
                value=4500.0, 
                min_value=1000.0, 
                max_value=10000.0,
                step=100.0,
                help="Cost per cubic meter of brickwork"
            )
            plaster_rate = st.number_input(
                "Plaster Rate (₹/m²)", 
                value=350.0, 
                min_value=100.0, 
                max_value=1000.0,
                step=10.0,
                help="Cost per square meter of plaster"
            )
        
        # Wall settings
        with st.expander("🧱 **Wall Settings**", expanded=True):
            default_thickness = st.selectbox(
                "Default Wall Thickness (mm)", 
                [115, 230, 345, 450], 
                index=1,
                help="Standard wall thickness for walls without explicit thickness"
            )
            wall_height = st.number_input(
                "Wall Height (mm)", 
                value=3000.0, 
                min_value=1000.0, 
                max_value=5000.0,
                step=100.0,
                help="Assumed height for all walls"
            )
        
        # Visualization settings
        with st.expander("🎨 **Visualization**", expanded=False):
            show_labels = st.checkbox("Show Wall Labels", value=True)
            show_3d = st.checkbox("Show 3D View", value=True)
            show_dashboard = st.checkbox("Show Dashboard", value=True)
        
        # Process button
        process_btn = st.button(
            "🚀 **Process DXF**", 
            type="primary", 
            use_container_width=True,
            disabled=uploaded_file is None
        )
        
        if uploaded_file is None:
            st.info("👈 **Upload a DXF file to begin**")
    
    with col_main:
        if uploaded_file is not None and process_btn:
            with st.spinner("🔄 **Processing DXF file...**"):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_path = tmp_file.name
                    
                    # Initialize variables
                    walls = []
                    boq_data = []
                    boq_with_costs = []
                    
                    # Process DXF
                    processor = DXFProcessor()
                    walls = processor.process_dxf(temp_path, default_thickness=default_thickness)
                    
                    # Check if walls were extracted
                    if not walls or len(walls) == 0:
                        st.error("❌ **No walls found in the DXF file.**")
                        st.info("""
                        **Possible reasons:**
                        1. The drawing doesn't contain LINE, LWPOLYLINE, or POLYLINE entities
                        2. Walls might be on non-standard layers
                        3. Drawing units might need adjustment
                        4. Try adjusting the default thickness setting
                        
                        **Try:**
                        - Open the DXF in a CAD viewer to check content
                        - Ensure walls are drawn as lines or polylines
                        - Check if walls are on visible layers
                        """)
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        return
                    
                    # Generate BOQ
                    generator = BOQGenerator()
                    boq_data = generator.generate_boq(walls, wall_height)
                    
                    # Calculate costs
                    boq_with_costs = generator.calculate_costs(boq_data, brick_rate, plaster_rate)
                    
                    # Display metrics in cards
                    st.markdown("### 📊 **Extraction Results**")
                    
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    
                    total_walls = len(walls)
                    total_length = sum(w.get('length', 0) for w in walls) / 1000  # Convert to meters
                    
                    # Calculate area from BOQ data
                    total_area = 0
                    if boq_data:
                        for item in boq_data:
                            if 'Area' in item['item']:
                                total_area += item.get('quantity', 0)
                    
                    total_cost = 0
                    if boq_with_costs:
                        total_cost = sum(item.get('total_cost', 0) for item in boq_with_costs)
                    
                    with col_metric1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Walls Extracted</div>
                            <div class="metric-value">{total_walls}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_metric2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Total Length</div>
                            <div class="metric-value">{total_length:.1f}<span class="metric-unit">m</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_metric3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Total Area</div>
                            <div class="metric-value">{total_area:.1f}<span class="metric-unit">m²</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_metric4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Estimated Cost</div>
                            <div class="metric-value">₹{total_cost:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualization Tabs
                    st.markdown("### 📐 **Visualizations**")
                    
                    viz_tabs = st.tabs(["2D Layout", "3D View", "Quantities", "Dashboard"])
                    
                    with viz_tabs[0]:
                        # 2D Visualization
                        fig_2d = visualizer.create_wall_visualization(walls, show_labels=show_labels)
                        st.pyplot(fig_2d)
                        
                        # Download button for the plot
                        buf = get_matplotlib_fig_bytes(fig_2d)
                        col_dl1, col_dl2 = st.columns([1, 4])
                        with col_dl1:
                            st.download_button(
                                label="📥 Download 2D Plot",
                                data=buf,
                                file_name=f"{uploaded_file.name.split('.')[0]}_2d_layout.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                    with viz_tabs[1]:
                        if show_3d:
                            # 3D Visualization
                            fig_3d = visualizer.create_3d_visualization(walls, wall_height)
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.info("Enable 3D visualization in the sidebar settings")
                    
                    with viz_tabs[2]:
                        # Quantity charts
                        fig_qty = visualizer.create_quantity_chart(boq_data)
                        st.pyplot(fig_qty)
                    
                    with viz_tabs[3]:
                        if show_dashboard:
                            # Comprehensive dashboard
                            fig_dash = visualizer.create_summary_dashboard(
                                walls, boq_data, uploaded_file.name
                            )
                            st.pyplot(fig_dash)
                        else:
                            st.info("Enable dashboard in the sidebar settings")
                    
                    # BOQ Data Section
                    st.markdown("### 📋 **Bill of Quantities**")
                    
                    # Display as dataframe
                    df_boq = pd.DataFrame(boq_with_costs)
                    st.dataframe(
                        df_boq,
                        use_container_width=True,
                        column_config={
                            "quantity": st.column_config.NumberColumn(format="%.3f"),
                            "rate": st.column_config.NumberColumn(
                                "Rate (₹)",
                                format="₹%.2f"
                            ),
                            "total_cost": st.column_config.NumberColumn(
                                "Total (₹)",
                                format="₹%.2f"
                            )
                        },
                        hide_index=True
                    )
                    
                    # Export options
                    st.markdown("### 📤 **Export Options**")
                    
                    col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
                    
                    with col_exp1:
                        # Excel export
                        excel_buffer = generator.export_to_excel(boq_with_costs)
                        st.download_button(
                            label="📊 Excel",
                            data=excel_buffer,
                            file_name=f"{uploaded_file.name.split('.')[0]}_boq.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download BOQ in Excel format",
                            use_container_width=True
                        )
                    
                    with col_exp2:
                        # JSON export
                        json_str = generator.export_to_json(boq_with_costs)
                        st.download_button(
                            label="📄 JSON",
                            data=json_str,
                            file_name=f"{uploaded_file.name.split('.')[0]}_boq.json",
                            mime="application/json",
                            help="Download BOQ in JSON format",
                            use_container_width=True
                        )
                    
                    with col_exp3:
                        # CSV export
                        csv_buffer = BytesIO()
                        df_boq.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        st.download_button(
                            label="📈 CSV",
                            data=csv_buffer,
                            file_name=f"{uploaded_file.name.split('.')[0]}_boq.csv",
                            mime="text/csv",
                            help="Download BOQ in CSV format",
                            use_container_width=True
                        )
                    
                    with col_exp4:
                        # Summary PDF
                        if st.button("📋 PDF Summary", use_container_width=True):
                            st.info("PDF export coming in next update!")
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Success message
                    st.success("✅ **Processing completed successfully!**")
                    
                except Exception as e:
                    st.error(f"❌ **Error processing file:** {str(e)}")
                    st.exception(e)
                    
                    # Clean up temp file if it exists
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.remove(temp_path)
        
        elif uploaded_file and not process_btn:
            # Show file preview before processing
            st.info("📁 **File ready for processing**")
            st.markdown(f"""
            **File Name:** {uploaded_file.name}
            
            **Click the 'Process DXF' button** in the sidebar to begin extraction.
            
            The system will:
            1. Extract all walls from the drawing
            2. Calculate lengths, areas, and volumes
            3. Generate Bill of Quantities
            4. Create visualizations
            5. Calculate estimated costs
            """)
            
            # Show sample visualization
            st.markdown("### 🎨 **Sample Visualization**")
            col_sample1, col_sample2 = st.columns(2)
            with col_sample1:
                st.image("https://via.placeholder.com/400x300/3498db/ffffff?text=2D+Wall+Layout", 
                        caption="2D Wall Layout")
            with col_sample2:
                st.image("https://via.placeholder.com/400x300/2ecc71/ffffff?text=3D+Visualization", 
                        caption="3D Visualization")
        
        else:
            # Welcome screen
            st.markdown("""
            ## 🏗️ **Welcome to DXF BOQ Extractor Pro**
            
            Transform your architectural DXF drawings into detailed **Bill of Quantities** with automatic calculations and visualizations.
            
            ### **How it works:**
            1. **Upload** your DXF drawing file
            2. **Configure** material rates and wall settings
            3. **Process** the drawing automatically
            4. **Review** extracted quantities and visualizations
            5. **Export** professional BOQ reports
            
            ### **Key Features:**
            ✅ **Automatic wall detection** from DXF entities  
            ✅ **Accurate quantity calculations** (length, area, volume)  
            ✅ **Material cost estimation** with customizable rates  
            ✅ **Interactive 2D & 3D visualizations**  
            ✅ **Multiple export formats** (Excel, JSON, CSV)  
            ✅ **Professional BOQ generation**  
            
            ### **Supported DXF Entities:**
            - Lines (wall centerlines)
            - Polylines (LWPOLYLINE, POLYLINE)
            - Arcs and Circles
            - Text and MText (for dimensions)
            - Layers and Blocks
            
            **👈 Start by uploading a DXF file in the sidebar!**
            """)
            
            # Feature showcase
            col_feat1, col_feat2, col_feat3 = st.columns(3)
            with col_feat1:
                st.markdown("""
                ### 📏 **Precise Measurements**
                Accurate extraction of wall lengths, areas, and volumes with unit conversion.
                """)
            
            with col_feat2:
                st.markdown("""
                ### 💰 **Cost Estimation**
                Automatic cost calculation based on current material rates.
                """)
            
            with col_feat3:
                st.markdown("""
                ### 📊 **Visual Reports**
                Professional visualizations for presentations and reports.
                """)

def render_boq_comparator():
    """Render BOQ Comparison Tab"""
    comparator.render_ui()

def main():
    # Main header
    st.markdown('<h1 class="main-header">📐 DXF BOQ Extractor Pro</h1>', unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🏗️ DXF Processing", "⚖️ BOQ Comparison"])
    
    with tab1:
        render_dxf_processor()
    
    with tab2:
        render_boq_comparator()
    
    # Footer
    st.divider()
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        **📞 Support**
        
        Email: support@dxfboq.com  
        Phone: +91-XXXXXXXXXX
        """)
    
    with footer_col2:
        st.markdown("""
        **🔗 Links**
        
        [Documentation](https://docs.dxfboq.com) | 
        [GitHub](https://github.com/yourusername/dxf-boq-extractor) | 
        [Report Issue](https://github.com/yourusername/dxf-boq-extractor/issues)
        """)
    
    with footer_col3:
        st.markdown("""
        **ℹ️ About**
        
        Version: 2.0.0  
        Last Updated: 2024  
        License: MIT
        """)

if __name__ == "__main__":
    # Check if all required modules exist
    required_modules = ['dxf_processor', 'boq_generator', 'visualization', 'boq_comparator']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        st.error(f"❌ **Missing modules:** {', '.join(missing_modules)}")
        st.info("""
        Please ensure all required Python files are in the same directory:
        - dxf_processor.py
        - boq_generator.py
        - visualization.py
        - boq_comparator.py
        """)
    else:
        main()