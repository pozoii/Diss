
import numpy as np
import pandas as pd
from ball_on_plate.envs.ball_on_plate import BallOnPlateEnv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from joblib import dump

class BallPDController:

    def __init__(self,kp=20.0,kd=5.0,max_torque=20.0):

        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque

    def __call__(self, state):

        x = state[0]
        y = state[1]

        x_dot = state[2]
        y_dot = state[3]

        # Desired tilt correction
        roll_command = (self.kp*x + self.kd*x_dot)
        pitch_command = (self.kp*y + self.kd*y_dot)

        action = np.array([roll_command,pitch_command])

        return np.clip(action,-self.max_torque,self.max_torque)
        
controller = BallPDController(kp=2,kd=4,max_torque=10.0)   


def learn_plate_dynamics(num_episodes=500,max_steps=5000):

    env = BallOnPlateEnv(render_mode=None,ball=True)
    records=[]

    for episode in range(num_episodes):
        s_t, info = env.reset()

        for step in range(max_steps):

            a_t = controller(s_t)
            s_prime, reward, terminated, truncated, info = env.step(a_t)
            #env.render()

            record= {
                "episode": episode,
                "step": step,
                "dt": env.dt,

                # state
                "alpha": s_t[4],
                "beta": s_t[5],
                "alpha_dot": s_t[6],
                "beta_dot": s_t[7],

                # state_prime
                "alpha_prime": s_prime[4],
                "beta_prime": s_prime[5],
                "alpha_dot_prime": s_prime[6],
                "beta_dot_prime": s_prime[7],

                # action
                "roll_torque": a_t[0],
                "pitch_torque": a_t[1],

                # angle accelerations
                "alpha_ddot": (s_prime[6] - s_t[6]) / env.dt,
                "beta_ddot":  (s_prime[7] - s_t[7]) / env.dt
                }
            if terminated or truncated:
                break
  
            
            records.append(record)
            s_t = s_prime
    env.close()
    df= pd.DataFrame(records)
    return df


df = learn_plate_dynamics()

plt.hist(df["alpha"], bins=100)
plt.show()

plt.hist(df["beta"], bins=100)
plt.show()

limit = np.deg2rad(30)
margin = np.deg2rad(5)   # keep a 2° safety margin

df_lim = df[(np.abs(df["alpha"]) < limit - margin) &(np.abs(df["beta"])  < limit - margin)]
#-------------------------------------------------------------------------------------------------------------------------------------------------

X_roll = df_lim[["roll_torque","alpha", "alpha_dot"]]
y_roll = df_lim["alpha_ddot"]

roll_model = LinearRegression(fit_intercept=True)
roll_model.fit(X_roll, y_roll)

X_pitch = pd.DataFrame({"pitch_torque": df_lim["pitch_torque"],"beta" : df_lim["beta"], "beta_dot": df_lim["beta_dot"]})
y_pitch = df_lim["beta_ddot"]

pitch_model = LinearRegression(fit_intercept=True)
pitch_model.fit(X_pitch, y_pitch)

dump({"roll_model": roll_model,"pitch_model": pitch_model,},"ball_on_plate/dynamics/plate_models.joblib")

"""print("Roll dynamics")
print("----------------")
print(f"Intercept : {roll_model.intercept_:.6f}")
print(f"Torque    : {roll_model.coef_[0]:.6f}")
print(f"alpha     : {roll_model.coef_[1]:.6f}")
print(f"alpha_dot : {roll_model.coef_[2]:.6f}")
print(f"R²        : {roll_model.score(X_roll, y_roll):.6f}")

plt.scatter(roll_model.predict(X_roll),df_lim["alpha_ddot"],s=2)

lims=[df_lim["alpha_ddot"].min(),df_lim["alpha_ddot"].max()]

plt.plot(lims,lims,"--")
plt.xlabel("predicted")
plt.ylabel("real")
plt.title("Roll acceleration predictions vs real")
plt.show()"""
"""print("\nPitch dynamics")
print("----------------")
print(f"Intercept : {pitch_model.intercept_:.6f}")
print(f"Torque    : {pitch_model.coef_[0]:.6f}")
print(f"beta      : {pitch_model.coef_[1]:.6f}")
print(f"beta_dot  : {pitch_model.coef_[2]:.6f}")
print(f"R²        : {pitch_model.score(X_pitch, y_pitch):.6f}")



plt.scatter(pitch_model.predict(X_pitch),df_lim["beta_ddot"],s=2)

lims=[df_lim["beta_ddot"].min(),df_lim["beta_ddot"].max()]

plt.plot(lims,lims,"--")
plt.xlabel("predicted")
plt.ylabel("real")
plt.title("Pitch acceleration predictions vs real")
plt.show()"""

# ---------------------------------------
# Validate on full dataset with angle clipping
# ---------------------------------------

limit = np.deg2rad(30)

df_test = df.copy()

# ------------------
# Roll prediction
# ------------------

X_roll_full = df_test[["roll_torque", "alpha", "alpha_dot"]]

alpha_ddot_pred = roll_model.predict(X_roll_full)

# integrate
alpha_dot_pred = df_test["alpha_dot"] + alpha_ddot_pred * df_test["dt"]
alpha_pred = df_test["alpha"] + alpha_dot_pred * df_test["dt"]

# apply joint limits
alpha_pred = np.clip(alpha_pred, -limit, limit)

# stop velocity when hitting limits
alpha_dot_pred = np.where(((alpha_pred >= limit) & (alpha_dot_pred > 0)) |((alpha_pred <= -limit) & (alpha_dot_pred < 0)),0,alpha_dot_pred)

df_test["alpha_ddot_pred"] = alpha_ddot_pred
df_test["alpha_pred"] = alpha_pred
df_test["alpha_dot_pred"] = alpha_dot_pred


# ------------------
# Pitch prediction
# ------------------

X_pitch_full = df_test[["pitch_torque", "beta", "beta_dot"]]

beta_ddot_pred = pitch_model.predict(X_pitch_full)

beta_dot_pred = df_test["beta_dot"] + beta_ddot_pred * df_test["dt"]
beta_pred = df_test["beta"] + beta_dot_pred * df_test["dt"]

beta_pred = np.clip(beta_pred, -limit, limit)

beta_dot_pred = np.where(((beta_pred >= limit) & (beta_dot_pred > 0)) |((beta_pred <= -limit) & (beta_dot_pred < 0)),0,beta_dot_pred)

df_test["beta_ddot_pred"] = beta_ddot_pred
df_test["beta_pred"] = beta_pred
df_test["beta_dot_pred"] = beta_dot_pred


"""print("Roll angle error")
print("----------------")
print("Mean:", np.mean(abs(df_test["alpha_prime"] - df_test["alpha_pred"])))
print("Max :", np.max(abs(df_test["alpha_prime"] - df_test["alpha_pred"])))

print("\nPitch angle error")
print("----------------")
print("Mean:", np.mean(abs(df_test["beta_prime"] - df_test["beta_pred"])))
print("Max :", np.max(abs(df_test["beta_prime"] - df_test["beta_pred"])))


plt.figure(figsize=(6,6))
plt.scatter(df_test["alpha_ddot_pred"],df_test["alpha_ddot"],s=2,alpha=0.1)

lims=[max(df_test["alpha_ddot_pred"].quantile(0.05), df_test["alpha_ddot"].quantile(0.95)),min(df_test["alpha_ddot_pred"].quantile(0.05), df_test["alpha_ddot"].quantile(0.95))]
plt.xlim(lims)
plt.ylim(lims)

plt.plot(lims,lims,"--")
plt.xlabel("Predicted alpha acceleration")
plt.ylabel("MuJoCo alpha acceleration")
plt.title("Roll dynamics validation")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(df_test["beta_ddot_pred"],df_test["beta_ddot"],s=2,alpha=0.3)

lims=[max(df_test["beta_ddot_pred"].quantile(0.05), df_test["beta_ddot"].quantile(0.05)),min(df_test["beta_ddot_pred"].quantile(0.95), df_test["beta_ddot"].quantile(0.95))]
plt.xlim(lims)
plt.ylim(lims)

plt.plot(lims,lims,"--")
plt.xlabel("Predicted beta acceleration")
plt.ylabel("MuJoCo beta acceleration")
plt.title("Pitch dynamics validation")
plt.show()"""



