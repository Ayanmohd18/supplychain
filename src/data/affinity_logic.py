import pandas as pd
import numpy as np
import plotly.graph_objects as go

class AffinityEngine:
    """
    SOTA Feature: Inter-Product Relationships (GNN-Lite).
    Identifies demand transfer between SKUs.
    """
    def __init__(self, data=None):
        self.data = data

    def calculate_affinities(self, items_list):
        """
        Generates a relationship matrix between selected items.
        Returns 'Complements' (positive) and 'Substitutes' (negative).
        """
        # Mocking GNN Output for UI Demonstration
        # In production, this would be the output of an Adjacency Matrix
        np.random.seed(42)
        n = len(items_list)
        matrix = np.random.uniform(-0.8, 0.8, size=(n, n))
        np.fill_diagonal(matrix, 1.0)
        
        return pd.DataFrame(matrix, index=items_list, columns=items_list)

    def create_relationship_graph(self, items_list):
        """Creates a Plotly Network Graph showing SKU nodes and their links."""
        df = self.calculate_affinities(items_list)
        
        fig = go.Figure()
        
        # Add Nodes
        for i, item in enumerate(items_list):
            angle = 2 * np.pi * i / len(items_list)
            x, y = np.cos(angle), np.sin(angle)
            
            # Add Edges (only strong ones)
            for j, other in enumerate(items_list):
                if i != j and abs(df.iloc[i, j]) > 0.4:
                    other_angle = 2 * np.pi * j / len(items_list)
                    ox, oy = np.cos(other_angle), np.sin(other_angle)
                    color = "rgba(0, 255, 0, 0.4)" if df.iloc[i, j] > 0 else "rgba(255, 0, 0, 0.4)"
                    fig.add_trace(go.Scatter(
                        x=[x, ox], y=[y, oy],
                        mode='lines',
                        line=dict(color=color, width=abs(df.iloc[i, j])*5),
                        hoverinfo='none',
                        showlegend=False
                    ))

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=20, color='#F5F5F5', line=dict(width=2, color='#FF1C1C')),
                text=[item.split('_')[-1]], # Short name
                textposition="top center",
                name=item
            ))

        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
