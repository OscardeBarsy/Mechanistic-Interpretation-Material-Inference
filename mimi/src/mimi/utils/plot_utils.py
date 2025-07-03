import os
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_attn(data, label_token, save_path):
    """
    Plots the attention map and saves the plot to the specified path.

    Args:
        data (torch.Tensor): The attention data to be plotted.
        label_token (list): List of tokens for labeling.
        save_path (str): Path to save the plot image.
    """
    # Convert the data to numpy
    data = data.cpu().numpy()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create tick values
    tickvals = np.arange(len(label_token))

    # Create the plot
    fig = px.imshow(
        data,
        labels={"x": "Head", "y": "Layer"},
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0.0,
    )
    fig.update_layout(
        width=300,  
        height=300,  
        title_font_size=14, 
        xaxis_title=None, 
        yaxis_title=None,  
        margin=dict(l=10, r=10, t=10, b=10),  
        font=dict(
            size=12,
            color="black"
        ),
        coloraxis_colorbar=dict(
            title="𝑆",  
            thicknessmode="pixels", 
            thickness=10, 
            lenmode="fraction",
            len=1 
        )
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor='black'
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor='black'
    )

    # Save the plot to the specified path
    fig.write_image(save_path)
    print(f"Plot saved to {save_path}")
    fig.show()


def plot_residual(data, label_token, save_path):
    """
    Plots the residual data and saves the plot to the specified path.

    Args:
        data (torch.Tensor): The residual data to be plotted.
        label_token (list): List of tokens for labeling.
        save_path (str): Path to save the plot image.
    """
    # Convert the data to numpy
    data = data.cpu().numpy()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create tick values
    tickvals = np.arange(len(label_token))

    # Create the plot
    fig = px.imshow(
        data.T,  
        y=label_token,
        labels={"y": "Sequence Position", "x": "Layer"},  
        color_continuous_scale='RdBu',  
        color_continuous_midpoint=0.0,
    )
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=label_token,
            showticklabels=True,  
        ),
        xaxis=dict(
            tickmode="array",
            showticklabels=True 
        ),
        width=400, 
        height=300, 
        title_font_size=14, 
        xaxis_title=None, 
        yaxis_title=None, 
        margin=dict(l=10, r=10, t=20, b=20), 
        font=dict(
            size=12,
            color="black"
        ),
        coloraxis_colorbar=dict(
            title="𝑆", 
            thicknessmode="pixels",  
            thickness=10, 
            lenmode="fraction",
            len=1  
        )
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks="", 
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks="", 
    )

    # Save the plot to the specified path
    fig.write_image(save_path)
    print(f"Plot saved to {save_path}")
    fig.show()

def plot_ablation(s_clean_logit_diff, mean_s_scores, sf_mean_s_scores, save_path):
    """
    Plots the ablation results and saves the plot to the specified path.

    Args:
        s_clean_logit_diff (float): Baseline value for the ablation.
        mean_s_scores (list): List of mean scores for necessity.
        sf_mean_s_scores (list): List of mean scores for sufficiency.
        save_path (str): Path to save the plot image.
    """
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y1 = [i.cpu() for i in mean_s_scores]
    y2 = [i.cpu() for i in sf_mean_s_scores]
    baseline = s_clean_logit_diff
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels,
        y=y1,
        name='Necessity',
        line=dict(color='#bcbd22'),
        mode='lines+markers',
        line_width=3,
        marker=dict(symbol='circle', size=8),
    ))
    fig.add_trace(go.Scatter(
        x=labels,
        y=y2,
        name='Sufficiency',
        line=dict(color='#9467bd'),
        mode='lines+markers',
        line_width=3,
        marker=dict(symbol='square', size=8),
    ))
    fig.add_hline(y=baseline, line_color="#000000", line_width=3, line_dash="dash")
    fig.update_layout(
        title="",
        width=500,
        height=500,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor='white',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            font=dict(size=20)
        ),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')

    # Save the plot to the specified path
    fig.write_image(save_path)
    print(f"Plot saved to {save_path}")
    fig.show()

def plot_ablation_robust(y1, y2, y3, y4, baseline1, baseline2, save_path, title="Robustness Evaluation Result"):
    """
    Plots the robustness evaluation result and saves the plot to the specified path.

    Args:
        y1, y2, y3, y4 (list): Data series to be plotted.
        baseline1, baseline2 (float): Baseline values for the plots.
        save_path (str): Path to save the plot image.
        title (str): Title of the plot.
    """
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=y1, name='Necessity (N)', line=dict(color='#bcbd22', dash='solid', width=3), marker=dict(symbol='circle')))  # Blue solid
    fig.add_trace(go.Scatter(x=labels, y=y2, name='Sufficiency (N)', line=dict(color='#9467bd', dash='solid', width=3), marker=dict(symbol='square')))  # Blue dotted
    fig.add_hline(y=baseline1, line_color="#000000", line_width=3, line_dash="dash", name='Baseline')
    fig.add_trace(go.Scatter(x=labels, y=y3, name='Necessity (Q)', line=dict(color='#bcbd22', dash='dot', width=3), marker=dict(symbol='circle'), showlegend=True))  # Orange solid
    fig.add_trace(go.Scatter(x=labels, y=y4, name='Sufficiency (Q)', line=dict(color='#9467bd', dash='dot', width=3), marker=dict(symbol='square'), showlegend=True))  # Orange dotted

    fig.add_hline(y=baseline2, line_color="#000000", line_width=3, line_dash="dash")

    fig.update_layout(
        title=title,
        width=500,
        height=500,
        margin=dict(l=0, r=0, t=10, b=10), 
        plot_bgcolor='white',
        barmode='group',
        bargap=0.1, 
        bargroupgap=0.1,  
        legend=dict(
            x=0.02, 
            y=0.98, 
            bgcolor="rgba(255, 255, 255, 0.8)",
            font=dict(
                size=20
            )
        ),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')

    # Save the plot to the specified path
    fig.write_image(save_path)
    print(f"Plot saved to {save_path}")
    fig.show()

def plot_ablation_syllogisms(mood_dics, save_path, title="Circuit transferability for unconditionally valid categorical syllogism"):
    """
    Plots the syllogism data and saves the plot to the specified path.

    Args:
        mood_dics (list): List of mood dictionaries containing data for the plots.
        save_path (str): Path to save the plot image.
        title (str): Title of the plot.
    """
    moods_ordered_label = ['AII-3', 'IAI-3', 'IAI-4', 'AAA-1', 'EAE-1', 'EIO-4', 'EIO-3', 'AII-1', 'AOO-2', 'AEE-4', 'OAO-3', 'EIO-1', 'EIO-2', 'EAE-2', 'AEE-2']

    num_layers = len(mood_dics)
    metrics = ['SYM']
    row_titles = moods_ordered_label
    fig_width = 1000  
    fig_height = 800  

    fig = make_subplots(
        rows=3, cols=5,
        subplot_titles=row_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,  # Reduced vertical spacing
        horizontal_spacing=0.05  # Reduced horizontal spacing
    )
    colors = {'Necessity': '#bcbd22', 'Sufficiency': '#9467bd'}
    for index, mood_dic in enumerate(mood_dics):
        row = (index // 5) + 1
        col = (index % 5) + 1
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        for trace_name in ['Necessity', 'Sufficiency']:
            y = mood_dic['mean_scores'] if trace_name == 'Necessity' else mood_dic['sf_mean_score']
            showlegend = (row == 1 and col == 1)

            fig.add_trace(
                go.Scatter(
                    x=labels, y=y, name=trace_name,
                    line=dict(color=colors[trace_name], width=2),
                    showlegend=showlegend
                ),
                row=row, col=col
            )
        if row == 1 and col == 1:  # Only show legend once
            fig.add_hline(y=mood_dic['clean_logit_diff'], line_color="#000000", line_width=1,
                          line_dash="dash", row=row, col=col, name="Baseline")
        else:
            fig.add_hline(y=mood_dic['clean_logit_diff'], line_color="#000000", line_width=1,
                          line_dash="dash", row=row, col=col)
    for col in range(1, 6):
        fig.update_xaxes(title_text="Ablated (Added) Heads", row=3, col=col)

    for row in range(1, 4):
            fig.update_yaxes(title_text="Mean Scores", row=row, col=col)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=24)),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=20)),
        height=fig_height, width=fig_width,
        plot_bgcolor='white',
        font=dict(family="Arial", size=20)
    )

    fig.update_xaxes(
        showline=True, linewidth=1, linecolor='black', mirror=True, tickmode='linear', dtick=5,
        title_font=dict(size=14),
        tickfont=dict(size=12)
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor='black', mirror=True,
        title_font=dict(size=14),
        tickfont=dict(size=12)
    )

    # Save the plot to the specified path
    fig.write_image(save_path)
    print(f"Plot saved to {save_path}")
    fig.show()