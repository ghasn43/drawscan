#!/usr/bin/env python3
"""
DXF BOQ Extractor with Comparison Feature
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
import json
from datetime import datetime
import zipfile
import io
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dxf_processor import DXFProcessor
    from boq_generator import BOQGenerator
    from boq_comparator import BOQComparator
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Import error: {e}")
    IMPORT_SUCCESS = False

# Page config
st.set_page_config(
    page_title="DXF BOQ Comparator",
    page_icon="⚖️",
    layout="wide"
)

def main():
    st.title("⚖️ DXF BOQ Comparator")
    st.markdown("### Compare Extracted Quantities with Client BOQs")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "⚙️ Compare", "📊 Results"])
    
    with tab1:
        show_upload_section()
    
    with tab2:
        show_comparison_section()
    
    with tab3:
        show_results_section()

def show_upload_section():
    """Upload section for DXF and BOQ files"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 DXF Drawing Files")
        dxf_files = st.file_uploader(
            "Upload DXF Files",
            type=['dxf', 'DXF'],
            accept_multiple_files=True,
            key="dxf_uploader"
        )
        
        if dxf_files:
            st.success(f"✅ {len(dxf_files)} DXF file(s) uploaded")
            
            # Show DXF files
            with st.expander("View DXF Files", expanded=False):
                for file in dxf_files:
                    st.write(f"**{file.name}** ({file.size / 1024:.1f} KB)")
    
    with col2:
        st.subheader("📋 Client BOQ Files")
        boq_files = st.file_uploader(
            "Upload BOQ Files",
            type=['xlsx', 'xls', 'csv', 'json'],
            accept_multiple_files=True,
            key="boq_uploader"
        )
        
        if boq_files:
            st.success(f"✅ {len(boq_files)} BOQ file(s) uploaded")
            
            # Show BOQ files
            with st.expander("View BOQ Files", expanded=False):
                for file in boq_files:
                    st.write(f"**{file.name}** ({file.size / 1024:.1f} KB)")
                    
                    # Preview BOQ if Excel/CSV
                    if file.name.endswith(('.xlsx', '.xls', '.csv')):
                        try:
                            if file.name.endswith('.csv'):
                                df = pd.read_csv(file)
                            else:
                                df = pd.read_excel(file)
                            
                            if st.button(f"Preview {file.name}", key=f"preview_{file.name}"):
                                st.dataframe(df.head(), use_container_width=True)
                        except:
                            pass
    
    # Store in session state
    if dxf_files:
        st.session_state['dxf_files'] = dxf_files
    if boq_files:
        st.session_state['boq_files'] = boq_files

def show_comparison_section():
    """Comparison configuration section"""
    if 'dxf_files' not in st.session_state or not st.session_state['dxf_files']:
        st.info("👆 Please upload DXF files in the Upload tab first")
        return
    
    st.subheader("⚙️ Comparison Settings")
    
    # Tolerance settings
    col1, col2 = st.columns(2)
    
    with col1:
        tolerance = st.slider(
            "Acceptable Difference (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Percentage difference considered acceptable"
        )
        
        units = st.selectbox(
            "Drawing Units",
            ["m", "mm", "cm"],
            index=0
        )
    
    with col2:
        project_name = st.text_input(
            "Project Name",
            value="Comparison Project"
        )
        
        enable_ai_matching = st.checkbox(
            "Enable AI Matching",
            value=True,
            help="Use AI to match similar descriptions"
        )
    
    # Comparison button
    if st.button("🚀 Start Comparison", type="primary", use_container_width=True):
        if IMPORT_SUCCESS:
            run_comparison(tolerance, units, project_name)
        else:
            st.error("Required modules not loaded. Please check installation.")

def run_comparison(tolerance, units, project_name):
    """Run the comparison process"""
    # Initialize
    comparator = BOQComparator(tolerance_percentage=tolerance)
    
    # Process DXF files
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_extracted_items = []
    
    # Process each DXF
    dxf_files = st.session_state.get('dxf_files', [])
    for i, dxf_file in enumerate(dxf_files):
        status_text.text(f"Processing DXF: {dxf_file.name}...")
        progress_bar.progress((i) / (len(dxf_files) * 2))
        
        # Save and process DXF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp_file:
            tmp_file.write(dxf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Extract quantities
            processor = DXFProcessor(units=units)
            if processor.load_file(Path(tmp_path)):
                entities = processor.extract_entities()
                quantities = processor.calculate_quantities(entities)
                
                # Generate BOQ
                generator = BOQGenerator(project_name=project_name)
                boq_df = generator.generate_boq(quantities, processor)
                
                if not boq_df.empty:
                    all_extracted_items.append({
                        'filename': dxf_file.name,
                        'boq': boq_df
                    })
        
        finally:
            os.unlink(tmp_path)
    
    # Process BOQ files
    boq_files = st.session_state.get('boq_files', [])
    all_client_boqs = []
    
    for i, boq_file in enumerate(boq_files):
        status_text.text(f"Loading BOQ: {boq_file.name}...")
        progress_bar.progress((len(dxf_files) + i) / (len(dxf_files) + len(boq_files)))
        
        # Save and load BOQ
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(boq_file.name).suffix) as tmp_file:
            tmp_file.write(boq_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load client BOQ
            client_boq = comparator.load_client_boq(tmp_path)
            all_client_boqs.append({
                'filename': boq_file.name,
                'boq': client_boq
            })
        
        except Exception as e:
            st.error(f"Error loading {boq_file.name}: {e}")
        finally:
            os.unlink(tmp_path)
    
    # Compare if we have both
    if all_extracted_items and all_client_boqs:
        status_text.text("Comparing quantities...")
        
        # Combine extracted BOQs
        combined_extracted = pd.concat([item['boq'] for item in all_extracted_items], ignore_index=True)
        
        # Compare with each client BOQ
        comparison_results = {}
        
        for client_item in all_client_boqs:
            results = comparator.compare_boqs(combined_extracted, client_item['boq'])
            comparison_results[client_item['filename']] = {
                'results': results,
                'extracted_boq': combined_extracted,
                'client_boq': client_item['boq']
            }
        
        # Store results
        st.session_state['comparison_results'] = comparison_results
        st.session_state['comparator'] = comparator
        st.session_state['comparison_date'] = datetime.now()
        
        # Update progress
        progress_bar.progress(1.0)
        status_text.text("Comparison complete!")
        
        st.success(f"✅ Comparison completed for {len(comparison_results)} BOQ file(s)")
        
        # Show summary
        for filename, data in comparison_results.items():
            report = comparator.generate_comparison_report(data['results'])
            accuracy = report['summary']['accuracy_percentage']
            
            st.metric(
                f"Accuracy for {filename}",
                f"{accuracy:.1f}%",
                delta=f"{accuracy - 100:.1f}%" if accuracy != 100 else None
            )
    
    else:
        st.warning("Need both DXF and BOQ files for comparison")

def show_results_section():
    """Display comparison results"""
    if 'comparison_results' not in st.session_state:
        st.info("Run a comparison first in the Compare tab")
        return
    
    st.subheader("📊 Comparison Results")
    
    # File selector
    filenames = list(st.session_state['comparison_results'].keys())
    selected_file = st.selectbox("Select BOQ File to View", filenames)
    
    if selected_file:
        data = st.session_state['comparison_results'][selected_file]
        results = data['results']
        comparator = st.session_state.get('comparator')
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        matches = sum(1 for r in results if r.status == 'Match')
        mismatches = sum(1 for r in results if r.status == 'Mismatch')
        missing = sum(1 for r in results if r.status == 'Missing')
        extra = sum(1 for r in results if r.status == 'Extra')
        
        with col1:
            st.metric("Total Items", len(results))
        with col2:
            st.metric("Matches", matches)
        with col3:
            st.metric("Mismatches", mismatches)
        with col4:
            st.metric("Issues", missing + extra)
        
        # Results table
        st.subheader("📋 Detailed Comparison")
        
        # Convert to DataFrame for display
        results_df = pd.DataFrame([{
            'Item': r.item_no,
            'Status': r.status,
            'Extracted Description': r.extracted_description[:50] + '...' if len(r.extracted_description) > 50 else r.extracted_description,
            'Client Description': r.client_description[:50] + '...' if len(r.client_description) > 50 else r.client_description,
            'Extracted Qty': f"{r.extracted_quantity:.3f}",
            'Client Qty': f"{r.client_quantity:.3f}",
            'Diff %': f"{r.percentage_difference:+.1f}%",
            'Similarity': f"{r.description_similarity:.1%}",
            'Notes': r.notes
        } for r in results])
        
        # Color code by status
        def color_status(val):
            if val == 'Match':
                color = 'green'
            elif val == 'Partial Match':
                color = 'orange'
            elif val == 'Mismatch':
                color = 'red'
            elif val == 'Missing':
                color = 'blue'
            else:  # Extra
                color = 'purple'
            return f'color: {color}; font-weight: bold'
        
        styled_df = results_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Issues section
        issues = [r for r in results if r.status in ['Mismatch', 'Missing', 'Extra']]
        if issues:
            st.subheader("🚨 Issues Requiring Attention")
            
            for issue in issues[:10]:  # Show first 10 issues
                with st.expander(f"Item {issue.item_no}: {issue.status}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if issue.extracted_description:
                            st.write(f"**Extracted:** {issue.extracted_description}")
                            st.write(f"Quantity: {issue.extracted_quantity} {issue.extracted_unit}")
                    
                    with col2:
                        if issue.client_description:
                            st.write(f"**Client:** {issue.client_description}")
                            st.write(f"Quantity: {issue.client_quantity} {issue.client_unit}")
                    
                    st.write(f"**Difference:** {issue.quantity_difference:+.3f} ({issue.percentage_difference:+.1f}%)")
                    st.write(f"**Note:** {issue.notes}")
            
            if len(issues) > 10:
                st.info(f"... and {len(issues) - 10} more issues")
        
        # Download section
        st.subheader("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Excel download
            if comparator:
                excel_buffer = io.BytesIO()
                comparator.export_comparison_to_excel(results, excel_buffer)
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_buffer.getvalue(),
                    file_name=f"BOQ_Comparison_{selected_file.replace('.', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            # JSON download
            if comparator:
                report = comparator.generate_comparison_report(results)
                json_str = json.dumps(report, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download JSON Data",
                    data=json_str,
                    file_name=f"BOQ_Comparison_{selected_file.replace('.', '_')}.json",
                    mime="application/json"
                )
        
        with col3:
            # Summary PDF
            summary_text = f"""
            BOQ COMPARISON REPORT
            =====================
            
            File: {selected_file}
            Date: {st.session_state.get('comparison_date', datetime.now())}
            Total Items: {len(results)}
            
            SUMMARY:
            • Matches: {matches}
            • Mismatches: {mismatches}
            • Missing in Drawing: {missing}
            • Extra in Drawing: {extra}
            
            KEY ISSUES:
            """
            
            for issue in issues[:5]:
                summary_text += f"\n• Item {issue.item_no}: {issue.status} - {issue.notes}"
            
            st.download_button(
                label="📥 Download Summary",
                data=summary_text,
                file_name=f"BOQ_Summary_{selected_file.replace('.', '_')}.txt",
                mime="text/plain"
            )
        
        # Visual comparison
        st.subheader("📈 Visual Comparison")
        
        # Create comparison chart data
        comparison_data = []
        for r in results:
            if r.extracted_quantity > 0 and r.client_quantity > 0:
                comparison_data.append({
                    'Item': r.item_no,
                    'Extracted': r.extracted_quantity,
                    'Client': r.client_quantity,
                    'Difference': r.quantity_difference
                })
        
        if comparison_data:
            chart_df = pd.DataFrame(comparison_data).head(20)  # Limit to 20 items
            
            # Show as table
            st.dataframe(chart_df, use_container_width=True)

def show_demo_comparison():
    """Show a demo comparison for illustration"""
    st.info("🔍 **Comparison Demo**")
    
    demo_data = {
        'Item': ['Concrete in walls', 'Plastering', 'Doors', 'Windows', 'Flooring'],
        'Extracted Qty': [25.3, 152.5, 12, 8, 85.6],
        'Client Qty': [24.8, 150.0, 10, 8, 90.0],
        'Unit': ['m³', 'm²', 'nos', 'nos', 'm²'],
        'Status': ['✅ Match', '✅ Match', '⚠️ Mismatch', '✅ Match', '⚠️ Mismatch'],
        'Difference %': ['+2.0%', '+1.7%', '+20.0%', '0.0%', '-4.9%']
    }
    
    demo_df = pd.DataFrame(demo_data)
    st.table(demo_df)

if __name__ == "__main__":
    # Initialize session state
    if 'dxf_files' not in st.session_state:
        st.session_state['dxf_files'] = []
    if 'boq_files' not in st.session_state:
        st.session_state['boq_files'] = []
    if 'comparison_results' not in st.session_state:
        st.session_state['comparison_results'] = {}
    
    # Run app
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>⚖️ <b>DXF BOQ Comparator</b> | Professional Quantity Verification Tool</p>
        <p>Compare extracted quantities with client BOQs for accurate verification</p>
    </div>
    """, unsafe_allow_html=True)