import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .predictor import predict

PLOT_TEMPLATE = dict(
    paper_bgcolor="#0f1b33",
    plot_bgcolor="#0f1b33",
    font=dict(color="#e6edf6"),
    margin=dict(l=20,r=20,t=40,b=20)
)


def purity_gauge(value):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix':"%"},
        title={'text':"Distillate Purity"},
        gauge={
            'axis':{'range':[50,100]},
            'bar':{'color':"#22c55e"},
            'bgcolor':"#0f1b33"
        }
    ))

    fig.update_layout(**PLOT_TEMPLATE)

    return fig


def energy_gauge(value):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix':" MW"},
        title={'text':"Energy Consumption"},
        gauge={
            'axis':{'range':[0,20]},
            'bar':{'color':"#3b82f6"},
            'bgcolor':"#0f1b33"
        }
    ))

    fig.update_layout(**PLOT_TEMPLATE)

    return fig


def reflux_sensitivity(z,T,R,P,N,F):

    R_range=np.linspace(1.2,6,40)

    purity=[]
    energy=[]

    for r in R_range:
        p,b,e=predict(z,T,r,P,N,F)
        purity.append(p*100)
        energy.append(e/1000)

    fig=make_subplots(specs=[[{"secondary_y":True}]])

    fig.add_trace(
        go.Scatter(
            x=R_range,
            y=purity,
            line=dict(color="#f59e0b",width=3),
            name="Purity %"),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=R_range,
            y=energy,
            line=dict(color="#ef4444",dash="dot"),
            name="Energy MW"),
        secondary_y=True
    )

    fig.update_layout(
        **PLOT_TEMPLATE,
        title=dict(text="Reflux Ratio Sensitivity")
    )

    return fig


def purity_contour(z,T,R,P,N,F):

    z_vals=np.linspace(0.1,0.9,30)
    R_vals=np.linspace(1.2,6,30)

    Z=[]

    for r in R_vals:
        row=[]
        for zi in z_vals:
            p,b,e=predict(zi,T,r,P,N,F)
            row.append(p*100)
        Z.append(row)

    fig=go.Figure(data=
        go.Contour(
            x=z_vals,
            y=R_vals,
            z=Z,
            colorscale="YlOrBr"
        )
    )

    fig.add_trace(
    go.Scatter(
        x=[z],
        y=[R],
        mode="markers",
        marker=dict(size=12,color="#22c55e",symbol="x"),
        name="Current"
    )
)

    fig.update_layout(
    **PLOT_TEMPLATE,
    title=dict(text="Purity Contour Map"),
    xaxis_title="Feed Composition",
    yaxis_title="Reflux Ratio"
)

    return fig
    

def mccabe_thiele(z, purity, bottoms):

    import numpy as np
    import plotly.graph_objects as go

    x = np.linspace(0,1,200)

    alpha = 2.4
    y_eq = (alpha*x)/(1+(alpha-1)*x)

    fig = go.Figure()

    # equilibrium curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_eq,
            name="Equilibrium",
            line=dict(color="#f59e0b",width=3)
        )
    )

    # diagonal
    fig.add_trace(
        go.Scatter(
            x=[0,1],
            y=[0,1],
            name="y = x",
            line=dict(color="#94a3b8",dash="dot")
        )
    )

    # operating line
    R=2.5
    slope=R/(R+1)
    intercept=purity/(R+1)

    y_op = slope*x + intercept

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_op,
            name="Operating Line",
            line=dict(color="#22c55e",width=2)
        )
    )

    # key points
    fig.add_trace(
        go.Scatter(
            x=[bottoms,z,purity],
            y=[bottoms,z,purity],
            mode="markers+text",
            text=["xB","zF","xD"],
            textposition="top center",
            marker=dict(size=10,color="#3b82f6"),
            showlegend=False
        )
    )

    fig.update_layout(
        title=dict(text="McCabe–Thiele Diagram"),
        paper_bgcolor="#0f1b33",
        plot_bgcolor="#0f1b33",
        font=dict(color="#e6edf6"),
        margin=dict(l=20,r=20,t=40,b=20)
    )

    return fig


def tray_sensitivity(z,T,R,P,N,F):

    trays=list(range(10,41))

    purity=[]

    for t in trays:
        p,b,e=predict(z,T,R,P,t,F)
        purity.append(p*100)

    fig=go.Figure()

    fig.add_bar(
        x=trays,
        y=purity,
        marker_color="#f59e0b"
    )

    fig.update_layout(
        **PLOT_TEMPLATE,
        title=dict(text="Tray Count Sensitivity")
    )

    return fig


def pressure_sensitivity(z,T,R,P,N,F):

    P_vals=np.linspace(80,200,40)

    purity=[]

    for p in P_vals:
        pr,b,e=predict(z,T,R,p,N,F)
        purity.append(pr*100)

    fig=go.Figure()

    fig.add_scatter(
        x=P_vals,
        y=purity,
        line=dict(color="#3b82f6")
    )

    fig.update_layout(
        **PLOT_TEMPLATE,
        title=dict(text="Pressure Sensitivity")
    )

    return fig


def operating_radar(purity,energy,N,F,R):

    values=[
        purity,
        N/40,
        F/500,
        1-abs(R-3)/5,
        1-energy/60
    ]

    labels=[
        "Purity",
        "Efficiency",
        "Capacity",
        "Stability",
        "Economy"
    ]

    fig=go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        line_color="#f59e0b"
    ))

    fig.update_layout(
        **PLOT_TEMPLATE,
        polar=dict(radialaxis=dict(range=[0,1])),
        title=dict(text="Operating Fingerprint")
    )

    return fig