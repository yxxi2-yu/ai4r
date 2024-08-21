# Autonomous Driving Gymnasium

**NOTE:** This is a long documentation page, which is done to make it easier for doing a "ctrl+f" search of all the documentation relating to this gymnasium.

Contents of this page:

* [Key Details](#key-details) of the gym.
* [Overview](#overview) description of the gym.
* [Car Details](#car-details): provides descriptions, equations, and parameters for the model of a car.
* [Road Details](#road-details): provides descriptions, equations, and parameters for the road environment.
* [Gymnasium Details](#gymnasium-details): provides descriptions, equations, and parameters for the gymnasium class that brings together a car and a road.

The full table of contents of this page, with all sub-sections listed:

[[_TOC_]]



## Key details:

* **Action space:** A `Box` space with 2 elements, one for the drive command and one for the steering request command.

* **Observation space:** A `Dict` space with with many possible keys. When creating the environment, you provide parameters that specify which observation to include.

* **Import:** `import ai4rgym` and `env = gymnasium.make("ai4rgym/autonomous_driving_env")`



## Overview

The Autonomous Driving Gym could be better to think of as a "road following gym". This gym simulates a car driving around on a horizontal 2D plane (i.e., the ground). **The goal** is that the car that should follow a line on the ground. The line-to-follow can be thought of as either the center-line of the lane of a road, or as the lane marking that the car should drive on the left- or right- hand-side of.



## Car Details

The car is assumed to be able to move freely around of a horizontal 2D plane and the effects of suspension and weight shifting are neglected. Hence, the car is described by a bicycle model:

* At **slow speeds**, a kinematic bicycle model is used, i.e., no slip of the wheels.
* At **high speeds**, a dynamic bicycle model is used, i.e., the wheels can slip.
* At **intermediate speeds**, the previous two are mixed together.
* The range of speeds for these three operating regimes are specified by the parameters `v_transition_min` and `v_transition_max` ([further details below](#car-equations-of-motion)).

The model for the car is based on the article "AMZ Driverless: The Full Autonomous Racing System" ([link to publisher](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21977))([link to preprint](https://arxiv.org/abs/1905.05150)). Further details and equations are provided in the [equations of motion](#car-equations-of-motion) section below.



### Car Arguments

Prototype of creating a car:

```python
from ai4rgym.envs.bicycle_model_dynamic import BicycleModelDynamic
car = BicycleModelDynamic(model_parameters)
```

The single argument is `model_parameters` as a dictionary that MUST include the following keys (and associated values):

| Dictionary Key  |  Description  |  Units  |
| :---:           | :---          |  :---:  |
| `"Lf"`  |  The length from the center-of-gravity to the center of the front wheel. |  meters |
| `"Lr"`  |  The length from the center-of-gravity to the center of the rear wheel. |  meters |
| `"m"`   |  The mass of the vehicle. |  kilograms |
| `"Iz"`  |  The mass moment of inertia about the vertical axis (i.e., the axis coming out of the 2D plane). | kg m^2 |
| `"Cm"`  |  The motor constant that converts the drive command to force applied to the wheel. |  Newtons per "100% command" |
| `"Cd"`  |  The coefficient of aerodynamics drag that opposes the direction of motion. |  Newtons / (m/s)^2 |
| `"delta_offset"`        |  The fixed offset in the steering angle that models mis-alignment of the wheels. The true steering angle of the vehicle is the requested  steering angle plus the value of `delta_offset`. |  radians |
| `"delta_request_max"`   |  Upper limit of allowable steering angle requests (the lower limit is taken as the negative). |  radians |
| `"Ddelta_lower_limit"`  |  Lower limit of allowable rate-of-change of steering angle. |  radians / second |
| `"Ddelta_upper_limit"`  |  Upper limit of allowable rate-of-change of steering angle. |  radians / second |

The following parameters can **optionally** be specified in the `model_parameters` dictionary:

| Dictionary Key  |  Description  |  Units  |
| :---:           | :---          |  :---:  |
| `"v_transition_min"`  |  The velocity below which the equation-of-motion simulate a purely kinematic model. **(Default: 3.0)**  |  m/s |
| `"v_transition_max"`  |  The velocity above which the equation-of-motion simulate a purely dynamic model. **(Default: 5.0)**  |  m/s |
| `"body_len_f"`        |  The length of the vehicle's body panels from the center-of-gravity to the front bumper-bar.  |  meters |
| `"body_len_r"`        |  length of the vehicle's body panels from the center-of-gravity to the rear bumper-bar.  |  meters |
| `"body_width"`        |  width of the vehicle's body panels from left to right.  |  meters |

The Pacejka's tyre-formala coefficients can also be optionally specified in the `model_parameters` dictionary for the road conditions of dry, wet, snow, and ice. The following is for "dry" conditions, and the other conditions follow the same pattern:

| Dictionary Key  |  Description  |
| :---:           | :---          |
| `"Dp"`          |  Pacejka's tyre formala **"peak"** coefficient |
| `"Cp"`          |  Pacejka's tyre formala **"shape"** coefficient |
| `"Bp"`          |  Pacejka's tyre formala **"stiffness"** coefficient |
| `"Ep"`          |  Pacejka's tyre formala **"curvature"** coefficient |

The default value for the Pacejka's tire formala coefficients (peak, shape, stiffness, curvature) are taken from [here](https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/) and [here](https://au.mathworks.com/help/sdl/ref/tireroadinteractionmagicformula.html). The details are repeated here for convenience:

|       | Name      |   Typical range | Dry tarmac | Wet tarmac | Snow  | Ice   |
| :---: | :---      | :---:           | :---:      | :---:      | :---: | :---: |
| D     | Peak      |   0.1  -  1.9   |   1.0      |   0.82     |  0.3  |  0.1  |
| C*    | Shape     |   1.0  -  2.0   |   1.9      |   2.3      |  2.0  |  2.0  |
| B     | Stiffness |   4.0  - 12.0   |  10.0      |  12.0      |  5.0  |  4.0  |
| E     | Curvature | -10.0  -  1.0   |   0.97     |   1.0      |  1.0  |  1.0  |

* The table above gives typical values for longitudinal forces.
* Note for coefficient C: The Pacekja model specifies the shape coefficient as C=1.65 for the longitudinal force and C=1.3 for the lateral force.



### Car Equations of Motion

As mentioned above, the model for the car is based on the bicycle model in the article "AMZ Driverless: The Full Autonomous Racing System" ([link to publisher](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21977))([link to preprint](https://arxiv.org/abs/1905.05150)).

The state of the car is defined by:

| Symbol      | Class variable       | Description                                                                       | Units   |
| :---:       | :---                 | :---                                                                              | :---:   |
| $(p_{x,\mathrm{cg}},p_{y,\mathrm{cg}})$ | `self.px`, `self.py` | Position of the vehicle's center-of-gravity in the world-frame Cartesian coordinate directions | meters  |
| $\omega$    | `self.omega`         | Heading angle of the car, relative to the world-frame x-axis                      | radians |
| $v_x$       | `self.vx`            | Velocity in the direction of the car, i.e., velocity along the body-frame x-axis  | m/s     |
| $v_y$       | `self.vy`            | Velocity along the body-frame y-axis                                              | m/s     |
| $\delta$    | `self.delta`         | Steering angle of the front wheels, relative to the body-frame x-axis             | radians |

The actions of the car are defined by:

| Symbol             | Class variable | Description                                                                           | Units       |
| :---:              | :---           | :---                                                                                  | :---:       |
| $F_{\mathrm{cmd}}$ | `self._Fcmd`   | Drive force command in percent applied to the rear wheels, valid range is [-100,100] where negative means a force in the reverse direction | percent |
| $\Delta_\delta$    | `self.Ddelta`  | Rate-of-change of steering angle                                                      | radians/s   |

The parameters described in the [car argument section](#car-arguments) above have the same mathematical symbol as the variable name, i.e., `"Lf"`, `"Lr"`, `"m"`, `"Iz"`, `"Cm"`, `"Cd"` is written in maths as $L_f, L_r, m, I_z, C_m, C_d$.

The dynamic equations of motions are thus given by:

```math
\begin{align}
\dot{p}_{x,\mathrm{cg}}     &\,=\,  v_x \cos(\theta)  -  v_y \sin(\theta)
\\
\dot{p}_{y,\mathrm{cg}}     &\,=\,  v_x \sin(\theta)  +  v_y \cos(\theta)
\\
\dot{\theta}  &\,=\,  \omega
\\
\dot{v}_x     &\,=\,  \frac{1}{m}  \left(  F       - F_{y,f} \sin(\delta)  +  m \, v_y \, \omega \right)
\\
\dot{v}_y     &\,=\,  \frac{1}{m}  \left(  F_{y,r} + F_{y,f} \cos(\delta)  -  m \, v_x \, \omega \right)
\\
\dot{\omega}  &\,=\,  \frac{1}{I_z} \left( -F_{y,r} \, L_r  +  F_{y,f} \, L_f \cos(\delta) \right)
\\
\dot{\delta}  &\,=\,  \Delta_\delta
\end{align}
```

Where the main drive force is computed as:

```math
\begin{align}
F  &\,=\,  \frac{F_{\mathrm{cmd}}}{100.0} C_m  -  C_d \, v_x^2
\end{align}
```

And the lateral tire forces are computed based on the so-called slip angles (which are the alpha's in the following) are computed as:

```math
\begin{align}
\alpha_f  &\,=\,  \arctan\left( \frac{(v_y + L_f \omega)}{v_x} \right)  -  \delta
\\
\alpha_r  &\,=\,  \arctan\left( \frac{(v_y - L_f \omega)}{v_x} \right)
\\
F_{y,f}   &\,=\,  F_{z,f} \, D_p \sin\left( C \arctan( B \, \alpha_f - E) \right)
\\
F_{y,r}   &\,=\,  F_{z,r} \, D_p \sin\left( C \arctan( B \, \alpha_r - E) \right)
\end{align}
```

For the purpose of mixing a kinematic bicycle together with the dynamic bicycle model above, the following form of the kinematic bicycle model is used:

```math
\begin{align}
\dot{p}_{x,\mathrm{cg}}     &\,=\,  v_x \cos(\theta)  -  v_y \sin(\theta)
\\
\dot{p}_{y,\mathrm{cg}}     &\,=\,  v_x \sin(\theta)  +  v_y \cos(\theta)
\\
\dot{\theta}  &\,=\,  \omega
\\
\dot{v}_x     &\,=\,  \frac{1}{m}  F
\\
\dot{v}_y     &\,=\,  \left( \dot{\delta} \sec^2(\delta) v_x + \tan(\delta) \dot{v}_x \right) \frac{L_r}{L_r + L_f}
\\
\dot{\omega}  &\,=\,  \left( \dot{\delta} \sec^2(\delta) v_x + \tan(\delta) \dot{v}_x \right) \frac{1}{L_r + L_f}
\\
\dot{\delta}  &\,=\,  \Delta_\delta
\end{align}
```

This kinematic bicycle model and the dynamic bicycle model above are linearly mixed when the speed of the car is in the range [`v_transition_min`, `v_transition_max`].

**Note:** The kinematic bicycle model in the [equations of motion section above](#car-equations-of-motion) is a little different to the "typical" kinematic bicycle model because the both components of the velocity vector are retained for compatibility with the dynamic bicycle model and the equations enforce the kinematic assumption of no slip.



### Steering Request as the Action

As per the [key details above](#key-details), the requested steering angle is the steering action to the Gymnasium. However, as per the [equations of motion section above](#car-equations-of-motion), the input the model is the angular rate of the steering. The gymnasium adjusts the steering fastest allowable angular rate to reach the requested steering angle. As an equation:

```python
self._Ddelta = max( self.Ddelta_lower_limit , min( (delta_request - self.delta) / Ts , self.Ddelta_upper_limit ) )
```

where `Ts` is the time step for one call to the `step` function of the Gymnasium.


### Kinematic car model and continuous-time linearization

For the purpose of using a model to synthesize a policy, it can be beneficial to use a low-fidelity model with the least states, even if this requires neglecting some aspects relative to a higher-fidelity model. One basis for this is the "typical" kinematic bicycle equations-of-motion.

In contrast to the kinematic bicycle model in the [equations of motion section above](#car-equations-of-motion), the "typical" kinematic bicycle equations-of-motion, for the rear wheel position $(p_{x,\mathrm{r}},p_{y,\mathrm{r}})$, are given by:

```math
\begin{align}
\dot{p}_{x,\mathrm{r}}     &\,=\,  v_x \cos(\theta)
\\
\dot{p}_{y,\mathrm{r}}     &\,=\,  v_x \sin(\theta)
\\
\dot{\theta}  &\,=\,  v_x \tan(\delta) \, \frac{1}{L_r+L_f}
\\
\dot{v}_x     &\,=\,  \Delta_v
\\
\end{align}
```

With the steering angle and the rate-of-change of velocity (i.e., acceleration) as the inputs (i.e., $\delta,\,\Delta_v$), we choose the following:

* We keep the steering angle as the input to this model (i.e., the steering action) to avoid put the [steering request as the action (mentioned above)](steering-request-as-the-action) into the "simple" equations-of-motion for model-based policies.

* We put the "F=ma" type conversion from drive-command to acceleration into the "simple" equations-of-motion for model-based policies so that the policy computes a drive action that can be directly applied to the Gymnasium. Additionally, we neglect the force due to aerodynamic drag. Thus, the final equation becomes:

```math
\begin{align}
\dot{v}_x  &\,=\,  \frac{1}{m} \, \frac{C_m}{100.0} \, F_{\mathrm{cmd}}
\end{align}
```

A model-based policy for driving along the road line is simpler to design and synthesize when the kinematic bicycle equations-of-motion are described in curvilinear coordinates (i.e., described in a coordinate frame that moves along the line-to-follow, as opposed to being described in the fixed world-frame coordinates).

The curvilinear coordinates, i.e., the state, are described by:

| Symbol       | Description                                                                       | Units   |
| :---:        | :---                                                                              | :---:   |
| $s$          | Road progress, i.e., distance measured along the line-to-follow from the start to the closest point on the line to the car's Cartesian position  | meters |
| $d$          | Closest distance from the car's Cartesian position to the line-to-follow, i.e., this distance is measured orthogonal to the line-to-follow. A positive value means the car is on the right-hand-side of the road relative to the direction of increasing progress, and a negative value means the left-hand-side  | meters |
| $\mu$        | Heading angles of the car relative to the tangent direction of the line-to-follow at the closet point, i.e., the angle from the tangent line to the body-frame x-axis  | radians |
| $\kappa(s)$  | Curvature of the road at the road progress point given by the argument "s" | 1/m     |


Converting the "typical" kinematic bicycle equations-of-motion above to curvilinear coordinates yields:

```math
\begin{align}
\dot{s}      &\,=\,  v_x \cos(\mu) \, \frac{1}{1-d \, \kappa(s)}
\\
\dot{d}      &\,=\,  v_x \sin(\mu)
\\
\dot{\mu}    &\,=\,  v_x \tan(\delta) \, \frac{1}{L_r+L_f} \,-\, \kappa(s) \, v_x \, \cos(\mu) \, \frac{1}{1-d \, \kappa(s)}
\\
\dot{v}_x    &\,=\,  \frac{1}{m} \, \frac{C_m}{100.0} \, F_{\mathrm{cmd}}
\\
\end{align}
```

The "simplest" model-based policy to synthesize is for following a straight line, i.e., curvature equals zero at all point along the road ($\kappa(s)=0$). The kinematic bicycle equations-of-motion for following a straight line are thus given by:
```math
\begin{align}
\dot{s}      &\,=\,  v_x \cos(\mu)
\\
\dot{d}      &\,=\,  v_x \sin(\mu)
\\
\dot{\mu}    &\,=\,  v_x \tan(\delta) \, \frac{1}{L_r+L_f}
\\
\dot{v}_x    &\,=\,  \frac{1}{m} \, \frac{C_m}{100.0} \, F_{\mathrm{cmd}}
\\
\end{align}
```

We note importantly that this is functionally identical to the "original" equations-of-motion given at the start of this sub-section. The key intuition is that we could choose the world-frame x-axis to point in any direction (albeit a fixed direction, we still get to choose that fixed direction).

We choose to linearize this last equations-of-motion for the steady-state behaviour of following the straight-line at a constant velocity, i.e., using a "bar" notation for the fixed point of linearize we have:

```math
\bar{s}=0,\,\, \bar{d}=0,\,\, \bar{\mu}=0,\,\, \bar{v}_x\neq 0,\,\, \bar{d}=0,\,\, \bar{\Delta_v}=0.
```

Hence, the linearization yields the following equations-of-motions that are linear in the state and actions:

```math
\begin{align}
\begin{bmatrix} \dot{s} \\ \dot{d} \\ \dot{\mu} \\ \dot{v}_x \end{bmatrix}
\,=\,
\begin{bmatrix}
    0 & 0 & 0 & 1
    \\
    0 & 0 & \bar{v}_x & 0
    \\
    0 & 0 & 0 & 0
    \\
    0 & 0 & 0 & 0
\end{bmatrix}
\,
\begin{bmatrix} s \\ d \\ \mu \\ (v_x-\bar{v}_x) \end{bmatrix}
\,+\,
\begin{bmatrix}
    0 & 0
    \\
    0 & 0
    \\
    0 & \bar{v}_x \, \frac{1}{L_r+L_f}
    \\
    \frac{1}{m} \, \frac{C_m}{100.0} & 0
\end{bmatrix}
\,
\begin{bmatrix} F_{\mathrm{cmd}} \\ \delta \end{bmatrix}
\end{align}
```

When we look carefully at the structure of these linearized equations-of-motion, we observe that the equations for how the car moves along the straight-line-to-follow are separate from (and uninfluenced by) the equation for how the car moves perpendicular to the line. This implies that we can reasonably synthesize two separate policies:

* A "cruise control" policy for regulating the speed of the car along the direction of the straight-line-to-follow by adjusting the drive command.
* A "lane keeping" policy for regulating the position of the car relative-to the straight-line-to-follow by adjusting the steering angle.

Explicitly writing out these two separate linearized equations-of-motion we get:

```math
\begin{align}
\begin{bmatrix} \dot{s} \\ \dot{v}_x \end{bmatrix}
&\,=\,
\begin{bmatrix}
    0 & 1
    \\
    0 & 0
\end{bmatrix}
\,
\begin{bmatrix} s \\ v_x \end{bmatrix}
\,+\,
\begin{bmatrix}
    0
    \\
    \frac{1}{m} \, \frac{C_m}{100.0}
\end{bmatrix}
\,
\begin{bmatrix} F_{\mathrm{cmd}} \end{bmatrix}
\,-\,
\begin{bmatrix} \bar{v}_x \\ 0 \end{bmatrix}
\\
\begin{bmatrix} \dot{d} \\ \dot{\mu} \end{bmatrix}
&\,=\,
\begin{bmatrix}
    0 & \bar{v}_x
    \\
    0 & 0
\end{bmatrix}
\,
\begin{bmatrix} d \\ \mu \end{bmatrix}
\,+\,
\begin{bmatrix}
    0
    \\
    \bar{v}_x \, \frac{1}{L_r+L_f}
\end{bmatrix}
\,
\begin{bmatrix} \delta \end{bmatrix}
\end{align}
```

It is important to keep in mind the following line-of-reasoning and intuition:

* Linearization assumes the state and action remain close to the linearization values (i.e., the "bar" values).
* For driving along a straight line, this assumption should hold true for all linearization values **expect the linearization speed**, i.e., $\bar{v}_x$.
* If the "cruise control" policy is used to change the car to a new average speed (for example from 80 km/h to 100 km/h),
* Then the linearization speed should be adjusted accordingly.
* The linearization speed appears in the "lane keeping" equations-of-motion, where it is:
    * a linear factor between the steering angle action and the angular velocity of the car; and also
    * a linear factor between the car's angle state and the perpendicular position state.
* This matches the intuition that at higher speeds the turning (and sideways movements) of the car is more sensitive to adjustments of the steering angle. (In fact, the steering mechanism of real-world cars is make to account for this increased sensitivity at high speeds.)
* This line-of-reasoning informs us, as the policy designers, that we need to consider how the policy changes as a function of the (average) speed.



### Kinematic car model as a discrete-time linear model for designing model-based policies

The final equation in the previous sub-section is essentially two so-called "double integrator" equations-of-motion. This relative simple form of the equations of motion allows us to use the matrix exponential to compute the exact time-discretation for a zero-order hold of the action.

Letting $T_s$ denote the sample time for time-discretization, the discrete-time linear equations-of-motion for designing a model-based **"cruise control"** policy are given by:

```math
\begin{align}
\begin{bmatrix} s_{k+1} \\ v_{x,k+1} \end{bmatrix}
&\,=\,
\begin{bmatrix}
    1 & T_s
    \\
    0 & 1
\end{bmatrix}
\,
\begin{bmatrix} s_k \\ v_{x,k} \end{bmatrix}
\,+\,
\begin{bmatrix}
    \frac{1}{2} \, T_s^2 \, \frac{1}{m} \, \frac{C_m}{100.0}
    \\
    T_s \, \frac{1}{m} \, \frac{C_m}{100.0}
\end{bmatrix}
\,
\begin{bmatrix} F_{\mathrm{cmd},k} \end{bmatrix}
\,-\,
\begin{bmatrix} T_s \, \bar{v}_x \\ 0 \end{bmatrix}
\end{align}
```

And the discrete-time linear equations-of-motion for designing a model-based **"lane keeping"** policy are given by:

```math
\begin{align}
\begin{bmatrix} d_{k+1} \\ \mu_{k+1} \end{bmatrix}
&\,=\,
\begin{bmatrix}
    1 & T_s \, \bar{v}_x
    \\
    0 & 1
\end{bmatrix}
\,
\begin{bmatrix} d_k \\ \mu_k \end{bmatrix}
\,+\,
\begin{bmatrix}
    \frac{1}{2} \, T_s^2 \, \bar{v}_x^2 \, \frac{1}{L_r+L_f}
    \\
    T_s \, \bar{v}_x \, \frac{1}{L_r+L_f}
\end{bmatrix}
\,
\begin{bmatrix} \delta_k \end{bmatrix}
\end{align}
```

For convenience, we repeat here the definition of all the symbols appearing in these equation:

| Symbol               | Description                                                                       | Units   |
| :---:                | :---                                                                              | :---:   |
| $s_k$                | Road progress at time-step k, i.e., distance measured along the line-to-follow from the start to the closest point on the line to the car's Cartesian position  | meters |
| $d_k$                | Closest distance from the car's Cartesian position to the line-to-follow at time-step k, i.e., this distance is measured orthogonal to the line-to-follow. A positive value means the car is on the right-hand-side of the road relative to the direction of increasing progress, and a negative value means the left-hand-side     | meters |
| $\mu_k$              | Heading angles of the car relative to the tangent direction of the line-to-follow at the closet point at time-step k, i.e., the angle from the tangent line to the body-frame x-axis  | radians |
| $v_{x,k}$            | Velocity of the car in the direction of the body frame x-axis at time-step k, i.e., this is the velocity at the rear-wheels | m/s     |
| $F_{\mathrm{cmd},k}$ | Drive force command in percent applied to the rear wheels at time-step k, where negative means a force in the reverse direction | percent |
| $\delta_k$           | Steering angle of the front wheels, relative to the body-frame x-axis, at time-step k | radians   |
| $T_s$                | Duration between two subsequent discrete time-step, i.e., sample time for time-discretization | seconds |
| $L_f$                | The length from the center-of-gravity to the center of the front wheel. |  meters |
| $L_r$                | The length from the center-of-gravity to the center of the rear wheel. |  meters |
| $m$                  | The mass of the vehicle. |  kilograms |
| $C_m$                | The motor constant that converts the drive command to force applied to the wheel. |  Newtons per "100% command" |



### Car Example - Telsa Model 3

The following model parameters somewhat approximate a Telsa Model 3 based on data taken from [here](https://www.tesla.com/ownersmanual/model3/en_cn/GUID-E414862C-CFA1-4A0B-9548-BE21C32CAA58.html) and [here](https://www.tesla.com/sites/default/files/blog_attachments/the-slipperiest-car-on-the-road.pdf):

```python
bicycle_model_parameters = {
    "Lf" : 0.55*2.875,
    "Lr" : 0.45*2.875,
    "m"  : 2000.0,
    "Iz" : (1.0/12.0) * 2000.0 * (4.692**2+1.850**2),
    "Cm" : (1.0/100.0) * (1.0 * 400.0 * 9.0) / 0.2286,
    "Cd" : 0.5 * 0.24 * 2.2204 * 1.202,
    "delta_offset"       :   0.0 * np.pi/180,
    "delta_request_max"  :  45.0 * np.pi/180,
    "Ddelta_lower_limit" : -45.0 * np.pi/180,
    "Ddelta_upper_limit" :  45.0 * np.pi/180,
}
```

Some notes on the data extracted from the sources and the conversion to the parameters above:

* **Weight and size:**
    * Wheel base = 2.875 [meters]
    * Mass (empty)  = { 1779 (f:845, r:934 ), 1900 (f:948, r:952 ), 1897 (f:946, r:951 ), 1617 (f:750, r:867 ) } [kg]
    * Mass (loaded) = { 2184 (f:975, r:1209), 2300 (f:1075,r:1225), 2300 (f:1075,r:1225), 2017 (f:878, r:1139) } [kg]
    * Length overall = 4.692 [m]
    * Width (w/o mirrors) = 1.850 [m]
    * &#2234 Mass moment of inertia (Iz) approx. = (1/12) * mass * width^2 * length^2

* **Drive force:**
    * Max torque (per motor) = {219, 326, 404} [Nm]
    * Motor speed at max torque = {6380, 6000, 5000} [RPM]
    * Gearbox ratio: 1:9
    * Number of motors = 2
    * Wheel diameter = {18, 19} [inches], i.e., radius = {0.2286, 0.2413} [meters]
    * &#2234 "Cm" = (1/100) * "max torque" / "wheel radius"

* **Aerodynamic drag:**
    * Drag coefficient = {0.24, 0.26}
    * Frontal drag area = {23.9, 25.2} [ft^2], conversion 0.3048^2 [m^2 / ft^2], thus equals {2.2204, 2.3412} [m^2]
    * Air density = 1.202 [kg/m^3]
    * Drag force = 0.5 * Cd * Area * rho_air * v^2
    * &#2234 "Cd" = 0.5 * Cd * Area * rho_air


### Functions in the Car Class

The following is a collection of a few functions from the `BicycleModelDynamic` class that may occasionally be useful, see the comments in the source code for documentation of these functions:

* `def reset(self, px = 0, py = 0, theta = 0, vx = 0, vy = 0, omega = 0, delta = 0)`
* `def set_action_requests(self, drive_command_request = 0.0, delta_request = 0.0)`
* `def perform_integration_step(self, Ts, method, num_steps = 1, should_update_state = False, road_condition = None, Dp=None, Cp=None, Bp=None, Ep=None)`
* `def get_actions(self)`
* `def render_car(self, axis_handle, px, py, theta, delta, scale=1.0, plot_handles=None)`
* ``



## Road Details

**NOTE:** The formal definition of **curvature** for a curved line is: `curvature = 1 / radius`. Hence it can be more intuitive to express all curvature in code values as the inverse of the radius. The main benefit of using curvature is that a straight line has zero curvature (and infinite radius).

The `Road` class defines the line-to-follow and provides function for getting details of the road relative to any coordinate in Cartesian space. The road is defined as a sequence of **elements** that smoothly join together (smooth in the sense of matching tangents at each join). The elements of the road can be either **straight** or **curved**, where a curved element is a circular arc.


### Road Arguments

Prototype of creating a road:

```python
from ai4rgym.envs.road import Road
road = Road(
    epsilon_c: float = (1.0/10000.0),
    road_elements_list: list = None,
)
```

The arguments are:

* `epsilon_c`  :  This is the minimum curvature that is allowed. Any curved road elements that are requested with a curvature less that the value of `epsilon_c` is set to have a curvature equal to `epsilon_c`. This minimum is enforced because some of the computations related to curved elements require division by the curvature, hence a minimum curvature ensures that numerical errors are avoided in such divisions.

* `road_elements_list`  :  This is a list of dictionaries where each element in the list provides the details for defining a straight element of road or a curved segment of road. The road (i.e., the line-to-follow) is constructed by sequentially joining together the elements (straight or curved) that are defined in the list. The following are examples of the three allowed dictionaries for defining a element:

    * `{"type":"straight", "length":3.0}`

    * `{"type":"curved", "curvature":1.0/50.0, "angle_in_degrees":45.0}`

    * `{"type":"curved", "curvature":1.0/50.0, "length":30.0}`

**Notes** for setting the values of:

* `"length"`  :  Must be a positive number. Units are meters. For a curved element, the `"length"` defines the arc-length of the element, which can be a more natural specification for low curvature and less natural for high curvature.

* `"curvature"`  :  Can be a positive or negative number. A positive curvature defines an element that curves to the left (counter-clockwise) in the direction of the road, and a negative curvature defines an element that curves to the right (clockwise). As mentioned above: "a curvature less that the value of `epsilon_c` is set to have a curvature equal to `epsilon_c`."

* `"angle_in_degrees"`  :  Must be a positive number. Units are degrees. This is only for curved elements and it describes the angle of the circles of the road element sweeps.



### Road Example

Based on the road arguments described above, the following is a complete example for defining a road with multiple elements:

```python
# Import the class
from ai4rgym.envs.road import Road

# Define the list of road elements
road_elements_list = [
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":1/100.0, "angle_in_degrees":180.0},
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":-1/50.0, "angle_in_degrees":180.0},
    {"type":"straight", "length":100.0},
]

# Create the "Road" object
road = Road(
    epsilon_c=(1.0/10000.0),
    road_elements_list=road_elements_list
)
```



### Functions in the Road Class

The following is a collection of a few functions from the road class that may occasionally be useful, see the comments in the source code for documentation of these functions:

* `def get_total_length(self)`
* `def add_road_element_straight(self, length=100)`
* `def add_road_element_curved_by_length(self, curvature=1/100, length=100)`
* `def add_road_element_curved_by_angle(self, curvature=1/100, angle_in_degrees=45)`
* `def render_road(self, axis_handle)`
* `def find_closest_point_to(self, px, py)`
* `def transform_points_2d( p , px_translate, py_translate, theta_rotate, p_translate_is_in_starting_frame=True)`
* `def road_info_at_given_pose_and_progress_queries(self, px, py, theta, progress_queries)`



## Gymnasium Details


The `AutonomousDrivingEnv` is a class that inherits from `gymnasium.Env`, hence it is a so-called "Gymnasium Environment". This Gymnasium simulates the interaction of:

* A car (as modelled by the [bicycle model described above](#car-details));
* That drives around on a road (as modelled by a sequence of straight lines and
      circlular arcs, which is the so-called [line-to-follow described above](#road-details)).

This Autonomous Driving Gymnasium environment contains a relatively large number of parameters that can be used to configure the observation space of the gymnasium without needing to add an observation wrapper. A reward wrapper is **always recommended** because the designer of a policy, and of the policy synthesis algorithm, is responsible for specifying a reward function that reflects their goals and performance metrics.


### Reward, truncation, and termination definitions

**NOTE:** re-iterating the statement from the [introduction to this section](#gymnasium-details), a reward wrapper is **always recommended** because the designer of a policy, and of the policy synthesis algorithm, is responsible for specifying a reward function that reflects their goals and performance metrics.

The "default" reward in the gymnasium implements the following equation:

```math
\begin{align}
\mathrm{reward}_k &\,=\, \left(s_k - s_{k-1}\right) \,-\, 10.0 \, d_k^2 \,-\, 5.0 \, \left(v_{x,k} - \bar{v}\right)^2,
\end{align}
```

where:

* The notation for the states $(s_k,\, d_k,\, v_{x,k})$ is defined in the [Car Details section above](#car-details), specifically in the [Car Equations of Motion sub-section above](#car-equations-of-motion) and in the [Kinematic car model and continuous-time linearization sub-section above](#kinematic-car-model-and-continuous-time-linearization).
* The fixed value of $\bar{v}$ is computed as the average of the initial velocity parameters described below.

The termination flag is raised and a termination reward added as per the following table. The parameters of the termination conditions and termination rewards can be specified via the [gymnasium arguments described below](#gymnasium-arguments).

| Termination condition | Termination reward  |
| :---                  | :---                |
| Car speed falls below a lower bound parameter | Fixed reward value given by the respective termination-reward parameter |
| Car speed goes above an upper bound parameter | Fixed reward value given by the respective termination-reward parameter |
| Distance from the car to the line-to-follow is further than an upper bound parameter (where the same upper bound is applied for both sides of the line-to-follow) | Fixed reward value given by the respective termination-reward parameter |


### Gymnasium Arguments

To be completed.