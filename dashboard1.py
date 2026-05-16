import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

#  Load & prep
import os
os.environ["MPLBACKEND"] = "Agg"

import requests
import io
import gc  # garbage collector

file_id = "13Ikzo0D-63clhK88URoF4y8ZUuKgjhNV"

print("Loading dataset...")
session = requests.Session()
response = session.get(
    "https://drive.google.com/uc",
    params={"export": "download", "id": file_id},
    stream=True
)
token = response.cookies.get("download_warning")
response = session.get(
    "https://drive.usercontent.google.com/download",
    params={"id": file_id, "export": "download", "confirm": token or "t"},
    stream=True
)

# Load only 50k rows, only the columns needed
COLS = [
    'age', 'gender', 'country', 'daily_usage_hours',
    'primary_platform', 'num_platforms_used', 'purpose',
    'avg_session_minutes', 'night_usage',
    'mental_health_score', 'addiction_level',
    'screen_time_before_sleep'
]

df = pd.read_csv(
    io.StringIO(response.content.decode("utf-8")),
    nrows=50000,
    usecols=COLS
)

# Free memory immediately
del response, session
gc.collect()

print(f"Loaded {len(df)} rows.")

df['age_group'] = pd.cut(
    df['age'],
    bins=[13, 18, 25, 35],
    labels=['Teen', 'Young Adult', 'Adult']
)

le = LabelEncoder()
df_model = df.copy()
for col in ['gender', 'country', 'primary_platform', 'purpose']:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

df_model['addiction_encoded'] = le.fit_transform(df_model['addiction_level'])

FEATURES = [
    'age', 'daily_usage_hours', 'num_platforms_used',
    'avg_session_minutes', 'mental_health_score',
    'screen_time_before_sleep', 'night_usage',
    'gender', 'primary_platform', 'purpose'
]

train_sample = df_model.sample(5000, random_state=42)
X = train_sample[FEATURES]
y = train_sample['addiction_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Done.")

importances = pd.DataFrame({
    'feature': FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

report = classification_report(
    y_test, rf.predict(X_test),
    target_names=['Low', 'Medium', 'High'],
    output_dict=True
)
report_df = pd.DataFrame(report).T.reset_index()
report_df.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
report_df = report_df[report_df['Class'].isin(['Low', 'Medium', 'High'])].round(3)

# Country-level aggregations
country_stats = df.groupby('country').agg(
    avg_mental_health=('mental_health_score', 'mean'),
    avg_daily_usage=('daily_usage_hours', 'mean'),
    avg_screen_time=('screen_time_before_sleep', 'mean'),
    total_users=('age', 'count'),
    night_usage_rate=('night_usage', 'mean')
).reset_index().round(2)

high_addiction = df[df['addiction_level'] == 'High'].groupby('country').size().reset_index(name='high_addiction_count')
total_per_country = df.groupby('country').size().reset_index(name='total')
high_addiction = high_addiction.merge(total_per_country, on='country')
high_addiction['high_addiction_pct'] = (high_addiction['high_addiction_count'] / high_addiction['total'] * 100).round(2)
country_stats = country_stats.merge(high_addiction[['country', 'high_addiction_pct']], on='country', how='left')

# App
app = dash.Dash(__name__)

HEADER_STYLE = {'textAlign': 'center', 'fontFamily': 'Arial', 'color': '#2c3e50'}
CARD_STYLE = {
    'backgroundColor': 'white', 'borderRadius': '8px',
    'padding': '15px', 'margin': '10px', 'boxShadow': '2px 2px 8px #ddd'
}

MAP_METRICS = {
    'avg_mental_health': 'Avg Mental Health Score',
    'avg_daily_usage': 'Avg Daily Usage Hours',
    'avg_screen_time': 'Avg Screen Time Before Sleep (mins)',
    'high_addiction_pct': '% High Addiction Users',
    'night_usage_rate': 'Night Usage Rate',
    'total_users': 'Total Users'
}

app.layout = html.Div([

    html.H1("Gen Z Social Media & Mental Health Dashboard", style=HEADER_STYLE),

    dcc.Tabs(id='tabs', value='eda', children=[

        # EDA Tab
        dcc.Tab(label='📊 Exploratory Analysis', value='eda', children=[

            html.Div([
                html.Div([
                    html.Label("Gender:"),
                    dcc.Dropdown(
                        id='gender-filter',
                        options=[{'label': g, 'value': g} for g in df['gender'].unique()],
                        multi=True, placeholder="All Genders"
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label("Addiction Level:"),
                    dcc.Dropdown(
                        id='addiction-filter',
                        options=[{'label': a, 'value': a} for a in ['Low', 'Medium', 'High']],
                        multi=True, placeholder="All Levels"
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label("Platform:"),
                    dcc.Dropdown(
                        id='platform-filter',
                        options=[{'label': p, 'value': p} for p in df['primary_platform'].unique()],
                        multi=True, placeholder="All Platforms"
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
            ], style={'padding': '20px'}),

            html.Div([
                html.Div([dcc.Graph(id='addiction-bar')], style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='platform-bar')], style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'padding': '10px'}),

            html.Div([
                html.Div([dcc.Graph(id='scatter-plot')], style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='boxplot')], style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'padding': '10px'}),

            html.Div([
                html.Div([dcc.Graph(id='age-group-bar')], style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='night-usage-bar')], style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'padding': '10px'}),

            html.Div([dcc.Graph(id='heatmap')], style=CARD_STYLE),
        ]),

        # ── Random Forest Tab ─────────────────────────────────────────────────
        dcc.Tab(label='🤖 Random Forest Model', value='rf', children=[

            html.Div([
                html.Div([
                    html.H3("Model Accuracy", style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.H2(f"{round(rf.score(X_test, y_test) * 100, 1)}%",
                            style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '48px'})
                ], style={**CARD_STYLE, 'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

                html.Div([
                    html.H3("Training Samples", style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.H2("40,000", style={'textAlign': 'center', 'color': '#2980b9', 'fontSize': '48px'})
                ], style={**CARD_STYLE, 'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

                html.Div([
                    html.H3("Test Samples", style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.H2("10,000", style={'textAlign': 'center', 'color': '#8e44ad', 'fontSize': '48px'})
                ], style={**CARD_STYLE, 'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

                html.Div([
                    html.H3("Features Used", style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.H2(str(len(FEATURES)),
                            style={'textAlign': 'center', 'color': '#e67e22', 'fontSize': '48px'})
                ], style={**CARD_STYLE, 'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
            ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '20px'}),

            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=px.bar(
                            importances, x='importance', y='feature',
                            orientation='h', color='importance',
                            color_continuous_scale='Blues',
                            title='Feature Importances'
                        ).update_layout(showlegend=False)
                    )
                ], style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("Classification Report", style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.Table(
                        [html.Tr([html.Th(col, style={
                            'padding': '10px', 'backgroundColor': '#2c3e50',
                            'color': 'white', 'textAlign': 'center'
                        }) for col in report_df.columns])] +
                        [html.Tr([
                            html.Td(report_df.iloc[i][col], style={
                                'padding': '10px', 'textAlign': 'center',
                                'backgroundColor': '#f8f9fa' if i % 2 == 0 else 'white'
                            }) for col in report_df.columns
                        ]) for i in range(len(report_df))],
                        style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'}
                    )
                ], style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'padding': '10px'}),

            html.Div([
                html.H3("🔮 Addiction Level Predictor", style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P("Adjust the sliders to predict addiction level for a user profile.",
                       style={'textAlign': 'center', 'color': '#7f8c8d'}),

                html.Div([
                    html.Div([
                        html.Label("Age:"),
                        dcc.Slider(13, 35, 1, value=20, id='sim-age',
                                   marks={13: '13', 25: '25', 35: '35'}, tooltip={"placement": "bottom"}),
                        html.Label("Daily Usage Hours:"),
                        dcc.Slider(0, 24, 0.5, value=4, id='sim-usage',
                                   marks={0: '0', 12: '12', 24: '24'}, tooltip={"placement": "bottom"}),
                        html.Label("Avg Session Minutes:"),
                        dcc.Slider(0, 300, 5, value=60, id='sim-session',
                                   marks={0: '0', 150: '150', 300: '300'}, tooltip={"placement": "bottom"}),
                    ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),

                    html.Div([
                        html.Label("Mental Health Score:"),
                        dcc.Slider(0, 10, 0.1, value=5, id='sim-mh',
                                   marks={0: '0', 5: '5', 10: '10'}, tooltip={"placement": "bottom"}),
                        html.Label("Screen Time Before Sleep (mins):"),
                        dcc.Slider(0, 120, 5, value=30, id='sim-screen',
                                   marks={0: '0', 60: '60', 120: '120'}, tooltip={"placement": "bottom"}),
                        html.Label("Night Usage:"),
                        dcc.RadioItems(id='sim-night',
                                       options=[{'label': ' No', 'value': 0}, {'label': ' Yes', 'value': 1}],
                                       value=0, inline=True),
                    ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),
                ]),

                html.Div(id='prediction-output', style={
                    'textAlign': 'center', 'fontSize': '28px',
                    'fontWeight': 'bold', 'padding': '20px', 'marginTop': '10px'
                })
            ], style=CARD_STYLE),
        ]),

        # Country Map Tab
        dcc.Tab(label='🌍 Country Map', value='map', children=[

            html.Div([
                html.Div([
                    html.Label("Select Metric to Map:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                    dcc.Dropdown(
                        id='map-metric',
                        options=[{'label': v, 'value': k} for k, v in MAP_METRICS.items()],
                        value='avg_mental_health',
                        clearable=False,
                        style={'width': '400px'}
                    )
                ], style={'padding': '20px'}),
            ]),

            # Choropleth map
            html.Div([dcc.Graph(id='choropleth-map')], style=CARD_STYLE),

            # Country stats row
            html.Div([
                html.Div([dcc.Graph(id='country-bar')],
                         style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='country-scatter')],
                         style={**CARD_STYLE, 'width': '47%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'padding': '10px'}),

            # Platform breakdown by country
            html.Div([dcc.Graph(id='country-platform')], style=CARD_STYLE),
        ]),
    ]),
], style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})


# EDA Callback
def filter_df(gender, addiction, platform):
    filtered = df.copy()
    if gender:
        filtered = filtered[filtered['gender'].isin(gender)]
    if addiction:
        filtered = filtered[filtered['addiction_level'].isin(addiction)]
    if platform:
        filtered = filtered[filtered['primary_platform'].isin(platform)]
    return filtered


@app.callback(
    Output('addiction-bar', 'figure'),
    Output('platform-bar', 'figure'),
    Output('scatter-plot', 'figure'),
    Output('boxplot', 'figure'),
    Output('age-group-bar', 'figure'),
    Output('night-usage-bar', 'figure'),
    Output('heatmap', 'figure'),
    Input('gender-filter', 'value'),
    Input('addiction-filter', 'value'),
    Input('platform-filter', 'value'),
)
def update_eda(gender, addiction, platform):
    filtered = filter_df(gender, addiction, platform)
    sample = filtered.sample(min(5000, len(filtered)), random_state=42)

    fig1 = px.histogram(filtered, x='addiction_level',
                        category_orders={'addiction_level': ['Low', 'Medium', 'High']},
                        color='addiction_level', title='Addiction Level Distribution')

    platform_means = filtered.groupby('primary_platform')['mental_health_score'].mean().reset_index()
    fig2 = px.bar(platform_means.sort_values('mental_health_score'),
                  x='primary_platform', y='mental_health_score',
                  color='mental_health_score', color_continuous_scale='Blues',
                  title='Avg Mental Health Score by Platform')

    fig3 = px.scatter(sample, x='daily_usage_hours', y='mental_health_score',
                      color='addiction_level',
                      category_orders={'addiction_level': ['Low', 'Medium', 'High']},
                      color_discrete_sequence=['green', 'orange', 'red'],
                      opacity=0.5, title='Daily Usage vs Mental Health')

    fig4 = px.box(sample, x='addiction_level', y='mental_health_score',
                  category_orders={'addiction_level': ['Low', 'Medium', 'High']},
                  color='addiction_level',
                  color_discrete_sequence=['green', 'orange', 'red'],
                  title='Mental Health Score by Addiction Level')

    age_means = filtered.groupby('age_group', observed=True)['mental_health_score'].mean().reset_index()
    fig5 = px.bar(age_means, x='age_group', y='mental_health_score',
                  color='age_group', title='Avg Mental Health Score by Age Group',
                  category_orders={'age_group': ['Teen', 'Young Adult', 'Adult']})

    night_means = filtered.groupby('night_usage')['mental_health_score'].mean().reset_index()
    night_means['night_usage'] = night_means['night_usage'].map({0: 'No Night Usage', 1: 'Night Usage'})
    fig6 = px.bar(night_means, x='night_usage', y='mental_health_score',
                  color='night_usage', color_discrete_sequence=['steelblue', 'tomato'],
                  title='Avg Mental Health Score by Night Usage')

    corr = filtered[['age', 'daily_usage_hours', 'num_platforms_used',
                      'avg_session_minutes', 'mental_health_score',
                      'screen_time_before_sleep']].corr()
    fig7 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                     title='Correlation Heatmap')

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7


# Predictor Callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('sim-age', 'value'),
    Input('sim-usage', 'value'),
    Input('sim-session', 'value'),
    Input('sim-mh', 'value'),
    Input('sim-screen', 'value'),
    Input('sim-night', 'value'),
)
def predict_addiction(age, usage, session, mh, screen, night):
    input_data = pd.DataFrame([{
        'age': age,
        'daily_usage_hours': usage,
        'num_platforms_used': 3,
        'avg_session_minutes': session,
        'mental_health_score': mh,
        'screen_time_before_sleep': screen,
        'night_usage': night,
        'gender': 1,
        'primary_platform': 2,
        'purpose': 1
    }])

    pred = rf.predict(input_data)[0]
    proba = rf.predict_proba(input_data)[0]
    label = ['Low', 'Medium', 'High'][pred]
    colors = {'Low': '#27ae60', 'Medium': '#e67e22', 'High': '#e74c3c'}
    proba_text = f"Low: {proba[0]:.1%}  |  Medium: {proba[1]:.1%}  |  High: {proba[2]:.1%}"

    return html.Div([
        html.Div(f"Predicted Addiction Level: {label}",
                 style={'color': colors[label], 'fontSize': '28px', 'fontWeight': 'bold'}),
        html.Div(proba_text, style={'color': '#7f8c8d', 'fontSize': '16px', 'marginTop': '10px'})
    ])


# Map Callbacks
@app.callback(
    Output('choropleth-map', 'figure'),
    Output('country-bar', 'figure'),
    Output('country-scatter', 'figure'),
    Output('country-platform', 'figure'),
    Input('map-metric', 'value'),
)
def update_map(metric):
    label = MAP_METRICS[metric]

    # Choropleth
    fig_map = px.choropleth(
        country_stats,
        locations='country',
        locationmode='country names',
        color=metric,
        hover_name='country',
        hover_data={
            'avg_mental_health': True,
            'avg_daily_usage': True,
            'high_addiction_pct': True,
            'total_users': True
        },
        color_continuous_scale='RdYlGn_r' if metric in ['high_addiction_pct', 'avg_daily_usage', 'avg_screen_time']
                               else 'Blues',
        title=f'World Map — {label}'
    )
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=500
    )

    # Top/bottom 10 bar chart
    top10 = country_stats.nlargest(10, metric)
    fig_bar = px.bar(
        top10.sort_values(metric),
        x=metric, y='country',
        orientation='h',
        color=metric,
        color_continuous_scale='Blues',
        title=f'Top 10 Countries — {label}'
    )
    fig_bar.update_layout(showlegend=False)

    # Scatter: daily usage vs mental health per country
    fig_scatter = px.scatter(
        country_stats,
        x='avg_daily_usage',
        y='avg_mental_health',
        size='total_users',
        color='high_addiction_pct',
        hover_name='country',
        color_continuous_scale='RdYlGn_r',
        title='Daily Usage vs Mental Health by Country',
        labels={
            'avg_daily_usage': 'Avg Daily Usage (hrs)',
            'avg_mental_health': 'Avg Mental Health Score',
            'high_addiction_pct': '% High Addiction'
        }
    )

    # Platform breakdown by top 10 countries
    top_countries = country_stats.nlargest(10, 'total_users')['country'].tolist()
    platform_country = (
        df[df['country'].isin(top_countries)]
        .groupby(['country', 'primary_platform'])
        .size()
        .reset_index(name='count')
    )
    fig_platform = px.bar(
        platform_country,
        x='country', y='count',
        color='primary_platform',
        barmode='group',
        title='Platform Usage by Top 10 Countries',
        labels={'count': 'Number of Users', 'primary_platform': 'Platform'}
    )

    return fig_map, fig_bar, fig_scatter, fig_platform


if __name__ == '__main__':
    app.run(debug=True)