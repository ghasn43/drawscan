"""
boq_comparator.py - Compare two BOQ Excel files for discrepancies
COMPLETE FIXED VERSION
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import difflib
import tempfile
import os

class BOQComparator:
    """Compare two BOQ Excel files and identify discrepancies"""
    
    def __init__(self, tolerance_percent: float = 5.0):
        """
        Args:
            tolerance_percent: Acceptable percentage difference (default 5%)
        """
        self.tolerance_percent = tolerance_percent
        self.comparison_results = None
        
    def load_boq_files(self, 
                      client_file_path: str, 
                      contractor_file_path: str,
                      client_sheet_name: Optional[str] = None,
                      contractor_sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load BOQ Excel files with intelligent column detection
        
        Args:
            client_file_path: Path to client's BOQ Excel file
            contractor_file_path: Path to contractor's BOQ Excel file
            client_sheet_name: Specific sheet name for client file
            contractor_sheet_name: Specific sheet name for contractor file
            
        Returns:
            Tuple of (client_df, contractor_df) DataFrames
        """
        # Try to auto-detect sheet names if not provided
        if client_sheet_name is None:
            client_sheet_name = self._detect_sheet_name(client_file_path)
        
        if contractor_sheet_name is None:
            contractor_sheet_name = self._detect_sheet_name(contractor_file_path)
        
        try:
            # Read Excel files
            client_df = pd.read_excel(client_file_path, sheet_name=client_sheet_name)
            contractor_df = pd.read_excel(contractor_file_path, sheet_name=contractor_sheet_name)
            
            # Standardize column names
            client_df = self._standardize_columns(client_df, "client")
            contractor_df = self._standardize_columns(contractor_df, "contractor")
            
            # Clean and preprocess data
            client_df = self._clean_dataframe(client_df)
            contractor_df = self._clean_dataframe(contractor_df)
            
            # Debug: Print column names
            print(f"Client columns after standardization: {client_df.columns.tolist()}")
            print(f"Contractor columns after standardization: {contractor_df.columns.tolist()}")
            
            return client_df, contractor_df
            
        except Exception as e:
            raise Exception(f"Error loading Excel files: {str(e)}")
    
    def _detect_sheet_name(self, file_path: str) -> str:
        """Detect the most likely sheet name containing BOQ data"""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Look for sheets with common BOQ-related names
            boq_keywords = ['boq', 'bill', 'quantities', 'schedule', 'takeoff', 'measurement']
            
            for sheet in sheet_names:
                if any(keyword in sheet.lower() for keyword in boq_keywords):
                    return sheet
            
            # If no matching sheet found, return first sheet
            return sheet_names[0]
        except Exception:
            return None
    
    def _standardize_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Standardize column names for comparison - ROBUST VERSION"""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Store original columns for debugging
        original_columns = df.columns.tolist()
        
        # Convert all column names to lowercase for easier matching
        df.columns = [str(col).strip() for col in df.columns]
        
        # Column mapping dictionary - expanded for better matching
        column_patterns = {
            'item': ['item', 'description', 'particulars', 'work', 'description of work',
                    'item desc', 'item description', 'work description', 'specification'],
            'quantity': ['quantity', 'qty', 'qty.', 'quantity (nos)', 'nos', 'number',
                        'count', 'unit quantity', 'quantity in', 'quantity of'],
            'unit': ['unit', 'units', 'unit of measurement', 'uom', 'unit of measure',
                    'measurement', 'unit type'],
            'rate': ['rate', 'unit rate', 'price', 'unit price', 'rate per unit',
                    'rate/unit', 'rate (rs.)', 'rate (₹)', 'rate per', 'unit cost'],
            'amount': ['amount', 'total', 'total amount', 'value', 'cost', 'total cost',
                      'amount (rs.)', 'amount (₹)', 'total value', 'sum', 'subtotal'],
            'remarks': ['remarks', 'notes', 'comments', 'observation', 'note', 'description']
        }
        
        # Function to check if a column name matches any pattern
        def matches_pattern(col_name, patterns):
            col_lower = str(col_name).lower()
            for pattern in patterns:
                if pattern in col_lower:
                    return True
            return False
        
        # Create mapping from original to standardized names
        column_mapping = {}
        used_patterns = set()
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Skip if already mapped
            if col in column_mapping:
                continue
            
            # Check each pattern
            for std_name, patterns in column_patterns.items():
                if std_name in used_patterns:
                    continue  # Don't reuse the same standard name
                    
                if matches_pattern(col_lower, patterns):
                    # Create the standardized column name
                    standardized_name = f"{std_name}_{source}"
                    column_mapping[col] = standardized_name
                    used_patterns.add(std_name)
                    break
        
        # If we found mappings, apply them
        if column_mapping:
            df.rename(columns=column_mapping, inplace=True)
        
        # Add source identifier
        df['source'] = source
        
        # Debug output
        print(f"Standardized {source} columns:")
        print(f"  Original: {original_columns}")
        print(f"  New: {df.columns.tolist()}")
        print(f"  Mapping: {column_mapping}")
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess DataFrame"""
        df = df.copy()
        
        # Remove completely empty rows and columns
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # Reset index after dropping rows
        df.reset_index(drop=True, inplace=True)
        
        # Identify potential numeric columns
        for col in df.columns:
            col_str = str(col).lower()
            
            # Skip source column
            if col == 'source':
                continue
                
            # Try to convert to numeric for quantity/rate/amount columns
            if any(keyword in col_str for keyword in ['quantity', 'qty', 'rate', 'price', 'amount', 'total', 'cost']):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    # If conversion fails, keep as is
                    pass
        
        # Fill NaN with appropriate values
        for col in df.columns:
            if col == 'source':
                continue
                
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')
        
        # Strip whitespace from string columns
        string_cols = df.select_dtypes(include=[object]).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def match_items(self, client_df: pd.DataFrame, contractor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match items between client and contractor BOQs using fuzzy matching
        
        Args:
            client_df: Client's BOQ DataFrame
            contractor_df: Contractor's BOQ DataFrame
            
        Returns:
            Merged DataFrame with matched items
        """
        # Debug: Show what we're working with
        print("\n" + "="*60)
        print("MATCHING ITEMS")
        print("="*60)
        print(f"Client DataFrame shape: {client_df.shape}")
        print(f"Contractor DataFrame shape: {contractor_df.shape}")
        print(f"Client columns: {client_df.columns.tolist()}")
        print(f"Contractor columns: {contractor_df.columns.tolist()}")
        
        # Find item columns
        client_item_col = None
        contractor_item_col = None
        
        for col in client_df.columns:
            if 'item_' in col:
                client_item_col = col
                break
        
        for col in contractor_df.columns:
            if 'item_' in col:
                contractor_item_col = col
                break
        
        if not client_item_col:
            # Try to find any column that might contain item descriptions
            for col in client_df.columns:
                if any(word in str(col).lower() for word in ['desc', 'particular', 'work']):
                    client_item_col = col
                    break
        
        if not contractor_item_col:
            # Try to find any column that might contain item descriptions
            for col in contractor_df.columns:
                if any(word in str(col).lower() for word in ['desc', 'particular', 'work']):
                    contractor_item_col = col
                    break
        
        print(f"Using client item column: {client_item_col}")
        print(f"Using contractor item column: {contractor_item_col}")
        
        # Extract item descriptions
        client_items = client_df[client_item_col].astype(str).tolist() if client_item_col else []
        contractor_items = contractor_df[contractor_item_col].astype(str).tolist() if contractor_item_col else []
        
        print(f"Client items: {len(client_items)}")
        print(f"Contractor items: {len(contractor_items)}")
        
        # Create a merged dataframe
        merged_data = []
        
        # First, try exact matches
        for idx, client_row in client_df.iterrows():
            client_desc = str(client_row.get(client_item_col, '')) if client_item_col else ''
            
            if not client_desc or client_desc.lower() in ['nan', 'none', '']:
                continue
            
            # Look for exact match
            exact_match = None
            if contractor_item_col:
                exact_match = contractor_df[
                    contractor_df[contractor_item_col].astype(str).str.lower() == client_desc.lower()
                ]
            
            if exact_match is not None and not exact_match.empty:
                for _, contractor_row in exact_match.iterrows():
                    merged_data.append(self._merge_rows(client_row, contractor_row, 'exact', 
                                                       client_item_col, contractor_item_col))
            else:
                # Try fuzzy matching
                best_match = self._fuzzy_match_item(client_desc, contractor_df, contractor_item_col)
                if best_match is not None:
                    merged_data.append(self._merge_rows(client_row, best_match, 'fuzzy',
                                                       client_item_col, contractor_item_col))
                else:
                    # No match found - client item only
                    merged_data.append(self._merge_rows(client_row, None, 'client_only',
                                                       client_item_col, contractor_item_col))
        
        # Add contractor items that weren't matched
        matched_contractor_items = []
        for row in merged_data:
            if contractor_item_col and contractor_item_col.replace('item_', 'item_contractor') in row:
                item = row.get(contractor_item_col.replace('item_', 'item_contractor'))
                if item:
                    matched_contractor_items.append(str(item))
        
        for _, contractor_row in contractor_df.iterrows():
            if not contractor_item_col:
                continue
                
            contractor_desc = str(contractor_row.get(contractor_item_col, ''))
            if contractor_desc and contractor_desc.lower() not in ['nan', 'none', '']:
                if contractor_desc not in matched_contractor_items:
                    merged_data.append(self._merge_rows(None, contractor_row, 'contractor_only',
                                                       client_item_col, contractor_item_col))
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(merged_data) if merged_data else pd.DataFrame()
        
        if comparison_df.empty:
            print("Warning: No items matched or compared")
            self.comparison_results = comparison_df
            return comparison_df
        
        print(f"\nComparison DataFrame created with {len(comparison_df)} rows")
        print(f"Columns: {comparison_df.columns.tolist()}")
        
        # Calculate differences if columns exist
        if 'quantity_client' in comparison_df.columns and 'quantity_contractor' in comparison_df.columns:
            comparison_df['quantity_diff'] = comparison_df['quantity_contractor'] - comparison_df['quantity_client']
            comparison_df['quantity_diff_percent'] = np.where(
                comparison_df['quantity_client'] != 0,
                (comparison_df['quantity_diff'] / comparison_df['quantity_client']) * 100,
                np.where(comparison_df['quantity_contractor'] != 0, 100, 0)
            )
        
        if 'rate_client' in comparison_df.columns and 'rate_contractor' in comparison_df.columns:
            comparison_df['rate_diff'] = comparison_df['rate_contractor'] - comparison_df['rate_client']
            comparison_df['rate_diff_percent'] = np.where(
                comparison_df['rate_client'] != 0,
                (comparison_df['rate_diff'] / comparison_df['rate_client']) * 100,
                np.where(comparison_df['rate_contractor'] != 0, 100, 0)
            )
        
        if 'amount_client' in comparison_df.columns and 'amount_contractor' in comparison_df.columns:
            comparison_df['amount_diff'] = comparison_df['amount_contractor'] - comparison_df['amount_client']
            comparison_df['amount_diff_percent'] = np.where(
                comparison_df['amount_client'] != 0,
                (comparison_df['amount_diff'] / comparison_df['amount_client']) * 100,
                np.where(comparison_df['amount_contractor'] != 0, 100, 0)
            )
        
        # Flag discrepancies - FIXED VERSION
        comparison_df['discrepancy_flag'] = 'No Issue'
        
        # Initialize masks as False Series with same index
        qty_mask = pd.Series([False] * len(comparison_df), index=comparison_df.index)
        rate_mask = pd.Series([False] * len(comparison_df), index=comparison_df.index)
        
        # Set masks based on available columns
        if 'quantity_diff_percent' in comparison_df.columns:
            qty_mask = abs(comparison_df['quantity_diff_percent']) > self.tolerance_percent
            comparison_df.loc[qty_mask, 'discrepancy_flag'] = 'Quantity Discrepancy'
        
        if 'rate_diff_percent' in comparison_df.columns:
            rate_mask = abs(comparison_df['rate_diff_percent']) > self.tolerance_percent
            comparison_df.loc[rate_mask, 'discrepancy_flag'] = 'Rate Discrepancy'
        
        # Check for both quantity and rate discrepancies
        both_mask = qty_mask & rate_mask
        if both_mask.any():
            comparison_df.loc[both_mask, 'discrepancy_flag'] = 'Major Discrepancy'
        
        # Flag unmatched items
        comparison_df.loc[comparison_df['match_type'] == 'client_only', 'discrepancy_flag'] = 'Client Item Only'
        comparison_df.loc[comparison_df['match_type'] == 'contractor_only', 'discrepancy_flag'] = 'Contractor Item Only'
        
        self.comparison_results = comparison_df
        
        print(f"Discrepancy distribution:")
        if 'discrepancy_flag' in comparison_df.columns:
            print(comparison_df['discrepancy_flag'].value_counts().to_dict())
        
        return comparison_df
    
    def _fuzzy_match_item(self, client_desc: str, contractor_df: pd.DataFrame, 
                         contractor_item_col: str) -> Optional[pd.Series]:
        """Find best fuzzy match for an item description"""
        if not contractor_item_col or contractor_df.empty:
            return None
        
        contractor_descs = contractor_df[contractor_item_col].astype(str).tolist()
        
        # Filter out empty descriptions
        contractor_descs = [desc for desc in contractor_descs if desc and desc.lower() not in ['nan', 'none', '']]
        
        if not contractor_descs:
            return None
        
        # Use difflib for fuzzy matching
        matches = difflib.get_close_matches(
            client_desc.lower(), 
            [desc.lower() for desc in contractor_descs],
            n=1,
            cutoff=0.6  # 60% similarity threshold
        )
        
        if matches:
            matched_desc = matches[0]
            # Find the index in the original list (case-insensitive)
            match_indices = [i for i, desc in enumerate(contractor_descs) 
                           if desc.lower() == matched_desc]
            
            if match_indices:
                # Get the actual index in the DataFrame
                actual_indices = contractor_df[
                    contractor_df[contractor_item_col].astype(str).str.lower() == matched_desc
                ].index.tolist()
                
                if actual_indices:
                    return contractor_df.loc[actual_indices[0]]
        
        return None
    
    def _merge_rows(self, client_row: Optional[pd.Series], 
                   contractor_row: Optional[pd.Series], 
                   match_type: str,
                   client_item_col: Optional[str] = None,
                   contractor_item_col: Optional[str] = None) -> Dict:
        """Merge client and contractor rows into a single dictionary"""
        merged = {'match_type': match_type}
        
        # Helper function to get column value with source suffix
        def get_value(row, col, source):
            if row is None or col not in row:
                return None
            return row[col]
        
        # Add client data
        if client_row is not None:
            for col in client_row.index:
                if col != 'source':
                    value = client_row[col]
                    # Add _client suffix to standard columns
                    if any(keyword in col for keyword in ['item_', 'quantity_', 'rate_', 'amount_', 'unit_']):
                        merged[f"{col}"] = value
                    else:
                        # For non-standard columns, add _client suffix
                        merged[f"{col}_client"] = value
        
        # Add contractor data
        if contractor_row is not None:
            for col in contractor_row.index:
                if col != 'source':
                    value = contractor_row[col]
                    # Add _contractor suffix to standard columns
                    if any(keyword in col for keyword in ['item_', 'quantity_', 'rate_', 'amount_', 'unit_']):
                        # Ensure we don't overwrite client columns
                        new_col = col.replace('item_', 'item_contractor') if 'item_' in col else col
                        merged[new_col] = value
                    else:
                        # For non-standard columns, add _contractor suffix
                        merged[f"{col}_contractor"] = value
        
        return merged
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of the comparison - ROBUST VERSION"""
        if self.comparison_results is None or self.comparison_results.empty:
            print("Warning: No comparison results available")
            return {}
        
        df = self.comparison_results
        
        # Initialize summary with defaults
        summary = {
            'total_items_client': 0,
            'total_items_contractor': 0,
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'client_only_items': 0,
            'contractor_only_items': 0,
            'total_discrepancies': 0,
            'quantity_discrepancies': 0,
            'rate_discrepancies': 0,
            'major_discrepancies': 0,
        }
        
        # Find client and contractor item columns
        client_item_cols = [col for col in df.columns if ('item_' in col and 'contractor' not in col)]
        contractor_item_cols = [col for col in df.columns if ('item_' in col and 'contractor' in col)]
        
        # Count client items
        if client_item_cols:
            client_item_col = client_item_cols[0]
            summary['total_items_client'] = len(df[df[client_item_col].notna()])
        
        # Count contractor items
        if contractor_item_cols:
            contractor_item_col = contractor_item_cols[0]
            summary['total_items_contractor'] = len(df[df[contractor_item_col].notna()])
        
        # Count match types
        if 'match_type' in df.columns:
            summary['exact_matches'] = len(df[df['match_type'] == 'exact'])
            summary['fuzzy_matches'] = len(df[df['match_type'] == 'fuzzy'])
            summary['client_only_items'] = len(df[df['match_type'] == 'client_only'])
            summary['contractor_only_items'] = len(df[df['match_type'] == 'contractor_only'])
        
        # Count discrepancies
        if 'discrepancy_flag' in df.columns:
            summary['total_discrepancies'] = len(df[df['discrepancy_flag'] != 'No Issue'])
            summary['quantity_discrepancies'] = len(df[df['discrepancy_flag'] == 'Quantity Discrepancy'])
            summary['rate_discrepancies'] = len(df[df['discrepancy_flag'] == 'Rate Discrepancy'])
            summary['major_discrepancies'] = len(df[df['discrepancy_flag'] == 'Major Discrepancy'])
        
        # Calculate financial summary
        client_amount_cols = [col for col in df.columns if ('amount_' in col and 'contractor' not in col)]
        contractor_amount_cols = [col for col in df.columns if ('amount_' in col and 'contractor' in col)]
        
        if client_amount_cols and contractor_amount_cols:
            client_amount_col = client_amount_cols[0]
            contractor_amount_col = contractor_amount_cols[0]
            
            total_client = df[client_amount_col].sum()
            total_contractor = df[contractor_amount_col].sum()
            
            summary['total_amount_client'] = total_client
            summary['total_amount_contractor'] = total_contractor
            summary['total_difference'] = total_contractor - total_client
            summary['percent_difference'] = (summary['total_difference'] / total_client * 100) if total_client != 0 else 0
        
        return summary
    
    def export_comparison_report(self, output_format: str = 'excel') -> BytesIO:
        """
        Export comparison results
        
        Args:
            output_format: 'excel', 'csv', or 'html'
            
        Returns:
            BytesIO buffer with the report
        """
        if self.comparison_results is None:
            raise ValueError("No comparison results available. Run match_items() first.")
        
        buffer = BytesIO()
        df = self.comparison_results
        
        if output_format.lower() == 'excel':
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Main comparison sheet
                df.to_excel(writer, sheet_name='BOQ Comparison', index=False)
                
                # Summary sheet
                summary = self.generate_summary_report()
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Discrepancies sheet
                if 'discrepancy_flag' in df.columns:
                    discrepancies = df[df['discrepancy_flag'] != 'No Issue']
                    if not discrepancies.empty:
                        discrepancies.to_excel(writer, sheet_name='Discrepancies', index=False)
                
                # Formatting
                workbook = writer.book
                
                # Add auto-filter to main sheet
                worksheet = writer.sheets['BOQ Comparison']
                worksheet.auto_filter.ref = worksheet.dimensions
                
            buffer.seek(0)
            
        elif output_format.lower() == 'csv':
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            
        elif output_format.lower() == 'html':
            html_content = df.to_html(index=False, classes='table table-striped')
            buffer.write(html_content.encode('utf-8'))
            buffer.seek(0)
        
        return buffer
    
    def create_visualizations(self):
        """Create Plotly visualizations for the comparison"""
        if self.comparison_results is None or self.comparison_results.empty:
            return {}
        
        df = self.comparison_results
        summary = self.generate_summary_report()
        
        visualizations = {}
        
        # 1. Match Type Distribution Pie Chart
        if 'match_type' in df.columns and not df['match_type'].empty:
            match_counts = df['match_type'].value_counts()
            if not match_counts.empty:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=match_counts.index,
                    values=match_counts.values,
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3)
                )])
                fig_pie.update_layout(
                    title='Item Matching Distribution',
                    height=400
                )
                visualizations['match_distribution'] = fig_pie
        
        # 2. Discrepancy Type Bar Chart
        if 'discrepancy_flag' in df.columns and not df['discrepancy_flag'].empty:
            disc_counts = df['discrepancy_flag'].value_counts()
            if not disc_counts.empty:
                fig_bar = go.Figure(data=[go.Bar(
                    x=disc_counts.index,
                    y=disc_counts.values,
                    marker_color=px.colors.sequential.Viridis
                )])
                fig_bar.update_layout(
                    title='Discrepancy Analysis',
                    xaxis_title='Discrepancy Type',
                    yaxis_title='Count',
                    height=400
                )
                visualizations['discrepancy_analysis'] = fig_bar
        
        # 3. Quantity Difference Scatter Plot
        if ('quantity_client' in df.columns and 'quantity_contractor' in df.columns and 
            'discrepancy_flag' in df.columns):
            # Filter out rows with NaN values
            valid_rows = df[df['quantity_client'].notna() & df['quantity_contractor'].notna()]
            if not valid_rows.empty:
                fig_scatter = px.scatter(
                    valid_rows,
                    x='quantity_client',
                    y='quantity_contractor',
                    color='discrepancy_flag',
                    hover_data=['item_client' if 'item_client' in df.columns else 'item_contractor'],
                    title='Quantity Comparison: Client vs Contractor',
                    labels={
                        'quantity_client': 'Client Quantity',
                        'quantity_contractor': 'Contractor Quantity'
                    }
                )
                # Add diagonal line for perfect match
                max_val = max(valid_rows['quantity_client'].max(), valid_rows['quantity_contractor'].max())
                fig_scatter.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Perfect Match',
                    line=dict(color='red', dash='dash')
                ))
                fig_scatter.update_layout(height=500)
                visualizations['quantity_comparison'] = fig_scatter
        
        # 4. Financial Summary Gauge Charts
        if ('total_amount_client' in summary and 'total_amount_contractor' in summary and
            summary['total_amount_client'] > 0):
            
            fig_gauges = go.Figure()
            
            # Client total gauge
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=summary['total_amount_client'],
                title={'text': "Client Total"},
                domain={'row': 0, 'column': 0},
                gauge={'axis': {'range': [None, summary['total_amount_client'] * 1.2]}}
            ))
            
            # Contractor total gauge
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=summary['total_amount_contractor'],
                title={'text': "Contractor Total"},
                domain={'row': 0, 'column': 1},
                gauge={'axis': {'range': [None, summary['total_amount_contractor'] * 1.2]}}
            ))
            
            # Difference gauge
            fig_gauges.add_trace(go.Indicator(
                mode="number+delta",
                value=summary['total_amount_contractor'],
                delta={'reference': summary['total_amount_client'], 'relative': True},
                title={'text': f"Difference: ₹{summary.get('total_difference', 0):,.2f}"},
                domain={'row': 0, 'column': 2}
            ))
            
            fig_gauges.update_layout(
                grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
                height=300,
                title='Financial Comparison'
            )
            visualizations['financial_gauges'] = fig_gauges
        
        return visualizations


class StreamlitBOQComparator:
    """Streamlit wrapper for BOQ Comparator"""
    
    def __init__(self):
        self.comparator = BOQComparator()
        self.comparison_df = None
        self.summary = None
        
    def render_ui(self):
        """Render the Streamlit UI for BOQ comparison"""
        st.title("⚖️ DXF BOQ Comparator")
        st.markdown("""
        Compare Client's BOQ with Contractor's BOQ to identify discrepancies in quantities, rates, and totals.
        """)
        
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Client's BOQ")
            client_file = st.file_uploader("Upload Client's Excel BOQ", 
                                          type=['xlsx', 'xls'], 
                                          key='client_boq')
            if client_file:
                st.success(f"✅ Client file uploaded: {client_file.name}")
        
        with col2:
            st.subheader("👷 Contractor's BOQ")
            contractor_file = st.file_uploader("Upload Contractor's Excel BOQ", 
                                              type=['xlsx', 'xls'], 
                                              key='contractor_boq')
            if contractor_file:
                st.success(f"✅ Contractor file uploaded: {contractor_file.name}")
        
        # Comparison settings
        with st.expander("⚙️ Comparison Settings", expanded=True):
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                tolerance = st.slider(
                    "Tolerance Percentage (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="Acceptable percentage difference before flagging as discrepancy"
                )
                self.comparator.tolerance_percent = tolerance
            
            with col_set2:
                sheet_option = st.radio(
                    "Sheet Selection",
                    ["Auto-detect", "Manual"],
                    help="Auto-detect finds BOQ sheet automatically"
                )
                
                if sheet_option == "Manual":
                    col_sheet1, col_sheet2 = st.columns(2)
                    with col_sheet1:
                        client_sheet = st.text_input("Client sheet name", value="")
                    with col_sheet2:
                        contractor_sheet = st.text_input("Contractor sheet name", value="")
                else:
                    client_sheet = None
                    contractor_sheet = None
        
        # Compare button
        compare_clicked = st.button("🔄 Compare BOQs", type="primary", use_container_width=True)
        
        if compare_clicked:
            if client_file and contractor_file:
                with st.spinner("Comparing BOQs..."):
                    try:
                        # Save uploaded files temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_client:
                            tmp_client.write(client_file.getbuffer())
                            client_path = tmp_client.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_contractor:
                            tmp_contractor.write(contractor_file.getbuffer())
                            contractor_path = tmp_contractor.name
                        
                        # Load and compare BOQs
                        client_df, contractor_df = self.comparator.load_boq_files(
                            client_path,
                            contractor_path,
                            client_sheet,
                            contractor_sheet
                        )
                        
                        # Show preview of loaded data
                        with st.expander("📊 Preview Loaded Data", expanded=False):
                            col_preview1, col_preview2 = st.columns(2)
                            with col_preview1:
                                st.write("**Client BOQ Preview:**")
                                st.dataframe(client_df.head(), use_container_width=True)
                            with col_preview2:
                                st.write("**Contractor BOQ Preview:**")
                                st.dataframe(contractor_df.head(), use_container_width=True)
                        
                        self.comparison_df = self.comparator.match_items(client_df, contractor_df)
                        self.summary = self.comparator.generate_summary_report()
                        
                        # Display results
                        self._display_results()
                        
                        # Clean up temp files
                        try:
                            os.unlink(client_path)
                            os.unlink(contractor_path)
                        except:
                            pass
                        
                    except Exception as e:
                        st.error(f"❌ Error comparing BOQs: {str(e)}")
                        st.exception(e)
                        
                        # Clean up temp files if they exist
                        try:
                            if 'client_path' in locals() and os.path.exists(client_path):
                                os.unlink(client_path)
                            if 'contractor_path' in locals() and os.path.exists(contractor_path):
                                os.unlink(contractor_path)
                        except:
                            pass
            else:
                st.warning("⚠️ Please upload both BOQ files to compare")
    
    def _display_results(self):
        """Display comparison results"""
        if self.comparison_df is None or self.summary is None:
            st.error("No comparison results available")
            return
        
        # Check if we have any data
        if self.comparison_df.empty:
            st.warning("No items to compare. The BOQ files might be empty or in an unexpected format.")
            return
        
        # Summary metrics
        st.subheader("📊 Comparison Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items Compared", 
                     self.summary.get('total_items_client', 0))
        
        with col2:
            st.metric("Exact Matches", 
                     self.summary.get('exact_matches', 0))
        
        with col3:
            discrepancies = self.summary.get('total_discrepancies', 0)
            st.metric("Discrepancies Found", 
                     discrepancies,
                     delta=f"-{discrepancies} issues" if discrepancies > 0 else "")
        
        with col4:
            if 'total_difference' in self.summary:
                diff = self.summary.get('total_difference', 0)
                percent = self.summary.get('percent_difference', 0)
                st.metric("Financial Difference", 
                         f"₹{diff:,.2f}",
                         delta=f"{percent:.1f}%")
        
        # Visualizations
        st.subheader("📈 Visual Analysis")
        
        visualizations = self.comparator.create_visualizations()
        
        if visualizations:
            # Create tabs for different visualizations
            tab_names = []
            if 'match_distribution' in visualizations:
                tab_names.append("📋 Match Distribution")
            if 'discrepancy_analysis' in visualizations:
                tab_names.append("⚠️ Discrepancies")
            if 'quantity_comparison' in visualizations:
                tab_names.append("📏 Quantity Comparison")
            if 'financial_gauges' in visualizations:
                tab_names.append("💰 Financial")
            
            if tab_names:
                tabs = st.tabs(tab_names)
                tab_index = 0
                
                if 'match_distribution' in visualizations:
                    with tabs[tab_index]:
                        st.plotly_chart(visualizations['match_distribution'], use_container_width=True)
                    tab_index += 1
                
                if 'discrepancy_analysis' in visualizations:
                    with tabs[tab_index]:
                        st.plotly_chart(visualizations['discrepancy_analysis'], use_container_width=True)
                    tab_index += 1
                
                if 'quantity_comparison' in visualizations:
                    with tabs[tab_index]:
                        st.plotly_chart(visualizations['quantity_comparison'], use_container_width=True)
                    tab_index += 1
                
                if 'financial_gauges' in visualizations:
                    with tabs[tab_index]:
                        st.plotly_chart(visualizations['financial_gauges'], use_container_width=True)
        else:
            st.info("No visualizations available for the current data.")
        
        # Detailed comparison table
        st.subheader("📋 Detailed Comparison")
        
        if not self.comparison_df.empty:
            # Filter options
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                show_discrepancies = st.checkbox("Show Only Discrepancies", value=False)
            
            with col_filter2:
                if 'match_type' in self.comparison_df.columns:
                    match_types = self.comparison_df['match_type'].unique().tolist()
                    match_type_filter = st.multiselect(
                        "Filter by Match Type",
                        options=match_types,
                        default=match_types
                    )
                else:
                    match_type_filter = []
                    st.info("No match type information available")
            
            with col_filter3:
                sort_options = ['match_type', 'discrepancy_flag']
                if 'quantity_diff_percent' in self.comparison_df.columns:
                    sort_options.append('quantity_diff_percent')
                if 'amount_diff' in self.comparison_df.columns:
                    sort_options.append('amount_diff')
                
                sort_by = st.selectbox(
                    "Sort By",
                    options=sort_options
                )
            
            # Apply filters
            filtered_df = self.comparison_df.copy()
            
            if show_discrepancies and 'discrepancy_flag' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['discrepancy_flag'] != 'No Issue']
            
            if match_type_filter and 'match_type' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['match_type'].isin(match_type_filter)]
            
            if sort_by in filtered_df.columns:
                # For percentage differences, sort by absolute value
                if 'percent' in sort_by.lower():
                    filtered_df = filtered_df.reindex(filtered_df[sort_by].abs().sort_values(ascending=False).index)
                else:
                    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
            
            # Display table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    'quantity_diff_percent': st.column_config.NumberColumn(
                        "Qty Diff %",
                        format="%.1f%%"
                    ) if 'quantity_diff_percent' in filtered_df.columns else None,
                    'amount_diff': st.column_config.NumberColumn(
                        "Amount Diff",
                        format="₹%.2f"
                    ) if 'amount_diff' in filtered_df.columns else None,
                    'rate_diff_percent': st.column_config.NumberColumn(
                        "Rate Diff %",
                        format="%.1f%%"
                    ) if 'rate_diff_percent' in filtered_df.columns else None,
                }
            )
            
            # Export options
            st.subheader("📤 Export Results")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                if st.button("📥 Download Excel Report", use_container_width=True):
                    try:
                        buffer = self.comparator.export_comparison_report('excel')
                        st.download_button(
                            label="Click to Download Excel",
                            data=buffer,
                            file_name=f"boq_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating Excel report: {e}")
            
            with col_exp2:
                if st.button("📥 Download CSV Report", use_container_width=True):
                    try:
                        buffer = self.comparator.export_comparison_report('csv')
                        st.download_button(
                            label="Click to Download CSV",
                            data=buffer,
                            file_name=f"boq_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating CSV report: {e}")
            
            with col_exp3:
                # Generate PDF report option
                if st.button("📄 Generate Summary Report", use_container_width=True):
                    st.info("Detailed PDF report generation feature coming soon!")
                    # Show a simple text summary
                    with st.expander("📝 Text Summary", expanded=False):
                        summary_text = f"""
                        BOQ Comparison Summary
                        ======================
                        
                        Client Items: {self.summary.get('total_items_client', 0)}
                        Contractor Items: {self.summary.get('total_items_contractor', 0)}
                        Exact Matches: {self.summary.get('exact_matches', 0)}
                        Fuzzy Matches: {self.summary.get('fuzzy_matches', 0)}
                        Client-Only Items: {self.summary.get('client_only_items', 0)}
                        Contractor-Only Items: {self.summary.get('contractor_only_items', 0)}
                        Total Discrepancies: {self.summary.get('total_discrepancies', 0)}
                        
                        """
                        if 'total_amount_client' in self.summary:
                            summary_text += f"""
                            Financial Summary:
                            ------------------
                            Client Total: ₹{self.summary.get('total_amount_client', 0):,.2f}
                            Contractor Total: ₹{self.summary.get('total_amount_contractor', 0):,.2f}
                            Difference: ₹{self.summary.get('total_difference', 0):,.2f}
                            Percentage Difference: {self.summary.get('percent_difference', 0):.1f}%
                            """
                        
                        st.text(summary_text)
            
            # Discrepancy resolution notes
            st.subheader("📝 Discrepancy Resolution Notes")
            
            with st.form("resolution_notes"):
                notes = st.text_area(
                    "Add notes about discrepancies for discussion with client/contractor:",
                    height=100,
                    placeholder="Note any issues, questions, or clarifications needed..."
                )
                
                submitted = st.form_submit_button("💾 Save Notes (Simulated)")
                if submitted:
                    if notes:
                        st.success("Notes saved! (In a real app, this would save to database or file)")
                    else:
                        st.warning("Please enter some notes before saving")
        else:
            st.warning("No comparison data available to display.")


# Test function
def test_comparator():
    """Test the BOQ comparator"""
    print("Testing BOQ Comparator...")
    
    # Create sample test files
    import pandas as pd
    
    # Create sample client BOQ
    client_data = {
        'Item Description': ['Brickwork', 'Plastering', 'Painting'],
        'Quantity': [10.5, 25.3, 15.8],
        'Unit': ['m³', 'm²', 'm²'],
        'Rate': [4500, 350, 180],
        'Amount': [47250, 8855, 2844]
    }
    
    # Create sample contractor BOQ with some differences
    contractor_data = {
        'Description of Work': ['Brickwork in foundation', 'Plastering work', 'Painting work', 'Extra Item'],
        'Qty': [11.2, 26.5, 15.8, 5.0],
        'UOM': ['m³', 'm²', 'm²', 'm'],
        'Unit Rate': [4600, 365, 185, 1200],
        'Total Amount': [51520, 9672.5, 2923, 6000]
    }
    
    client_df = pd.DataFrame(client_data)
    contractor_df = pd.DataFrame(contractor_data)
    
    # Save test files
    client_df.to_excel('test_client_boq.xlsx', index=False)
    contractor_df.to_excel('test_contractor_boq.xlsx', index=False)
    
    print("Created test files: test_client_boq.xlsx, test_contractor_boq.xlsx")
    
    # Test the comparator
    comparator = BOQComparator(tolerance_percent=5.0)
    
    try:
        # Load files
        client_df_loaded, contractor_df_loaded = comparator.load_boq_files(
            'test_client_boq.xlsx',
            'test_contractor_boq.xlsx'
        )
        
        print(f"Loaded client shape: {client_df_loaded.shape}")
        print(f"Loaded contractor shape: {contractor_df_loaded.shape}")
        
        # Match items
        comparison_df = comparator.match_items(client_df_loaded, contractor_df_loaded)
        
        print(f"Comparison shape: {comparison_df.shape if comparison_df is not None else 'None'}")
        
        # Generate summary
        summary = comparator.generate_summary_report()
        print("\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Test export
        excel_buffer = comparator.export_comparison_report('excel')
        print(f"\nExcel export successful: {len(excel_buffer.getvalue())} bytes")
        
        # Clean up test files
        import os
        if os.path.exists('test_client_boq.xlsx'):
            os.remove('test_client_boq.xlsx')
        if os.path.exists('test_contractor_boq.xlsx'):
            os.remove('test_contractor_boq.xlsx')
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    test_comparator()