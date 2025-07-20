import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_gauge_chart(probability, threshold=0.5):
    # Calculate the prediction strength text
    if probability > 0.8:
        strength = "Very Strong"
        color = '#4338ca'  # Deep purple for very strong
    elif probability > 0.6:
        strength = "Strong"
        color = '#6366f1'  # Purple for strong
    elif probability > 0.4:
        strength = "Moderate"
        color = '#f59e0b'  # Amber for moderate
    else:
        strength = "Weak"
        color = '#ef4444'  # Red for weak

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 28, 'color': '#f8fafc'}},  
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 50, 'increasing': {'color': '#4f46e5'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickmode': 'array',
                'tickvals': [0, 25, 50, 75, 100],
                'ticktext': ['0%', '25%', '50%', '75%', '100%'],
                'tickfont': {'size': 14, 'color': '#f8fafc'},  
                'tickcolor': '#f8fafc'  
            },
            'bar': {'color': '#f8fafc'},  
            'bgcolor': '#1e293b',  
            'borderwidth': 2,
            'bordercolor': '#475569',
            'steps': [
                {'range': [0, 25], 'color': '#dc2626', 'line': {'width': 1, 'color': '#475569'}},  # Strong red
                {'range': [25, 50], 'color': '#f59e0b', 'line': {'width': 1, 'color': '#475569'}}, # Strong amber
                {'range': [50, 75], 'color': '#6366f1', 'line': {'width': 1, 'color': '#475569'}}, # Strong purple
                {'range': [75, 100], 'color': '#4338ca', 'line': {'width': 1, 'color': '#475569'}} # Deep purple
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': threshold * 100
            }
        },
        title={
            'text': f"Prediction Confidence<br><sub>{strength}</sub>",
            'font': {'size': 20, 'color': '#f8fafc'}
        }
    ))
    
    # Add annotations with confidence level explanations
    annotations = [
        dict(
            x=0.1, y=0.85,
            text="<b>Very Strong</b><br>75-100%",
            showarrow=False,
            align='left',
            font={'size': 12, 'color': '#4338ca'}
        ),
        dict(
            x=0.1, y=0.65,
            text="<b>Strong</b><br>50-75%",
            showarrow=False,
            align='left',
            font={'size': 12, 'color': '#6366f1'}
        ),
        dict(
            x=0.1, y=0.45,
            text="<b>Moderate</b><br>25-50%",
            showarrow=False,
            align='left',
            font={'size': 12, 'color': '#f59e0b'}
        ),
        dict(
            x=0.1, y=0.25,
            text="<b>Weak</b><br>0-25%",
            showarrow=False,
            align='left',
            font={'size': 12, 'color': '#dc2626'}
        ),
        dict(
            x=0.5, y=-0.2,
            text="<b>50% Threshold:</b> Above = High Income (>$50K), Below = Moderate Income (≤$50K)",
            showarrow=False,
            font={'size': 12, 'color': '#94a3b8'}
        )
    ]
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e293b", 'family': "Arial"},
        annotations=annotations
    )
    return fig

def create_feature_importance_chart(importance_dict):
    df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker_color='#4f46e5'
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e293b", 'family': "Arial"},
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def create_income_distribution(user_value, feature_name, dataset):
    fig = go.Figure()
    
    # Add histogram for the feature distribution
    fig.add_trace(go.Histogram(
        x=dataset[feature_name],
        name='Distribution',
        marker_color='#4f46e5',
        opacity=0.7
    ))
    
    # Add vertical line for user's value
    fig.add_vline(
        x=user_value,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Your Value",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"{feature_name} Distribution",
        xaxis_title=feature_name,
        yaxis_title="Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e293b", 'family': "Arial"},
        showlegend=True
    )
    return fig

def create_radar_chart(user_data, average_data, feature_names):
    # Create more readable labels
    display_names = {
        'age': 'Age Range',
        'education-num': 'Education Level',
        'capital-gain': 'Capital Gains',
        'capital-loss': 'Capital Losses',
        'hours-per-week': 'Working Hours'
    }
    
    # Convert feature names to display names
    display_features = [display_names.get(f, f) for f in feature_names]
    
    fig = go.Figure()
    
    # Add trace for user's data
    fig.add_trace(go.Scatterpolar(
        r=user_data,
        theta=display_features,
        fill='toself',
        name='Your Profile',
        line_color='#4f46e5',
        fillcolor='rgba(99, 102, 241, 0.2)'
    ))
    
    # Add trace for average data
    fig.add_trace(go.Scatterpolar(
        r=average_data,
        theta=display_features,
        fill='toself',
        name='Average Profile',
        line_color='#ef4444',
        fillcolor='rgba(239, 68, 68, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont={'size': 10},
                ticksuffix='%',
                showline=True,
                linecolor='rgba(255, 255, 255, 0.1)',
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            angularaxis=dict(
                tickfont={'size': 12},
                rotation=90,
                direction="clockwise",
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0)',
            font={'size': 12}
        ),
        title={
            'text': "Your Profile vs. Average Profile",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f8fafc", 'family': "Arial"}
    )
    return fig
