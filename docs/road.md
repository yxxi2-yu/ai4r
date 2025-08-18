# Road Definition and Usage

This document explains how to define roads for the `ai4rgym` environment using the `Road` class in `ai4rgym/envs/road.py`, including speed limits and recommended speeds.

## Overview
- A road is a sequence of elements: straight lines and circular arcs.
- Base per-element properties:
  - `c` (curvature, 1/m) and `l` (length, m).
  - `v_max` (maximum speed), stored internally in m/s.
- Derived per-element properties include:
  - `phi` (angular span, radians), `isStraight` flag.
  - Start/end points and angles, arc center, accumulative length, hyperplanes.
  - `v_rec` (recommended speed), computed from curvature and `v_max`.

## Units
- Length: meters, angles: radians, curvature: 1/m.
- Speed inputs in APIs that construct elements use km/h; they are converted and stored internally in m/s.
- Getter methods return speeds in m/s.

## Recommended Speed
The recommended speed per element is computed as a simple comfort/limit heuristic:
- `v_rec = min(v_max, sqrt(a_lat_max / |curvature|))` for non‑zero curvature.
- `v_rec = v_max` for straight elements.
- `a_lat_max = 2.0 m/s^2` by default.

## Constructing Roads via a Spec List
You can pass a list of element dictionaries to the constructor via `road_elements_list`.

Supported element shapes:
- Straight: `{ "type": "straight", "length": <meters>, "v_max_kph": <km/h, optional> }`
- Curved (length specified): `{ "type": "curved", "curvature": <1/m>, "length": <meters>, "v_max_kph": <km/h, optional> }`
- Curved (angle specified): `{ "type": "curved", "curvature": <1/m>, "angle_in_degrees": <deg>, "v_max_kph": <km/h, optional> }`

Notes:
- If `v_max_kph` is not provided, it defaults to 100 km/h for that element.
- All speeds are stored internally in m/s; recommended speeds are computed automatically.

Example:
```python
from ai4rgym.envs.road import Road

road_spec = [
    {"type": "straight", "length": 50,  "v_max_kph": 80},
    {"type": "curved",   "curvature": 0.020, "length": 60,  "v_max_kph": 60},
    {"type": "curved",   "curvature": -0.015, "angle_in_degrees": 45, "v_max_kph": 50},
    {"type": "straight", "length": 100},  # defaults to 100 km/h
]

road = Road(road_elements_list=road_spec)
```

## Constructing Roads Programmatically
You can also add elements incrementally using methods that accept `v_max_kph` (default 100).

```python
from ai4rgym.envs.road import Road

road = Road()
road.add_road_element_straight(length=100, v_max_kph=90)
road.add_road_element_curved_by_length(curvature=0.010, length=80, v_max_kph=70)
road.add_road_element_curved_by_angle(curvature=-0.015, angle_in_degrees=30, v_max_kph=60)
```

## Accessing Geometry and Speeds
```python
# Geometry
c = road.get_c()                    # curvature per element (1/m)
l = road.get_l()                    # length per element (m)
phi = road.get_phi()                # angular span (rad)

# Speeds (m/s)
v_max_mps = road.get_v_max()        # per-element max speeds in m/s
v_rec_mps = road.get_v_recommended()# per-element recommended speeds in m/s

# Convert to km/h for display if desired
v_max_kph = v_max_mps * 3.6
v_rec_kph = v_rec_mps * 3.6
```

## Updating Max Speeds After Construction
The helper methods work in m/s. Convert from km/h if you are working in those units.

```python
# Set all elements at once (values in m/s)
road.set_max_speeds([27.78, 22.22, 19.44, 27.78])  # 100, 80, 70, 100 km/h

# Or set a single element (value in m/s)
road.set_max_speed_for_element(0, 25.0)  # ~90 km/h

# Recompute recommended speeds (usually called automatically by setters)
road.recompute_recommended_speeds()
```

## Tips
- Use `v_max_kph` in the constructor spec and adder methods; internal storage is m/s.
- Recommended speeds are updated automatically when elements are added or when you change `v_max` via the provided setters.
- For extreme curves, the recommended speed may be significantly lower than the maximum speed due to the lateral acceleration limit.

