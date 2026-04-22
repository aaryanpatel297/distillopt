import numpy as np
import pandas as pd

def derive_features(z,T_F,R,P,N_T,F):

    P_mmHg=P*7.50062

    A1,B1,C1=6.90565,1211.033,220.790
    A2,B2,C2=6.95334,1343.943,219.377

    T_bub=80.0

    for _ in range(50):
        P1=10**(A1-B1/(C1+T_bub))
        P2=10**(A2-B2/(C2+T_bub))

        f=z*P1+(1-z)*P2-P_mmHg

        df=z*P1*B1/(C1+T_bub)**2*np.log(10)+(1-z)*P2*B2/(C2+T_bub)**2*np.log(10)

        T_bub-=f/df

    P1_sat=10**(A1-B1/(C1+T_bub))
    P2_sat=10**(A2-B2/(C2+T_bub))

    alpha=P1_sat/P2_sat

    x_D=0.7+0.28*z
    x_B=0.3*z

    odds_D=x_D/(1-x_D)
    odds_B=x_B/(1-x_B)

    rel_vol=odds_D/odds_B

    reflux_feed=R/F

    sep_factor=(x_D-z)/(z-x_B+1e-6)

    tray_util=np.log(rel_vol)/np.log(N_T+1)

    pt_index=P*T_F

    return pd.DataFrame([{
        "feed_composition_molfrac":z,
        "feed_temperature_C":T_F,
        "reflux_ratio":R,
        "column_pressure_kPa":P,
        "num_trays":N_T,
        "feed_flow_rate_kmolph":F,
        "relative_volatility_est":rel_vol,
        "reflux_to_feed_ratio":reflux_feed,
        "separation_factor":sep_factor,
        "tray_utilisation":tray_util,
        "pressure_temp_index":pt_index
    }])