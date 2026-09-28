# Lighthouses — knowledge map

> How coastal lights are built, identified and run. The list of individual lights is
> `Lighthouse Inventory.md`. The navigator's use of them is `../navigation/Navigation.md` §4.

## 1. What a light is for

A light answers one of two questions, and the design follows from which. A **landfall light**
says "the coast is here", and wants maximum range. A **harbour light** says "the safe channel
is exactly here", and wants precision — a narrow sector, sharply bounded, where range beyond
the approach is of no value at all.

Confusing the two produces the worst of both: a bright light that does not tell you where the
channel is.

## 2. Optics

### 2.1 Reflector

A parabolic mirror behind a flame. Simple, and wasteful: a great deal of light escapes past
the mirror entirely. Superseded but still found in small harbour lights.

### 2.2 Fresnel lens

Concentric prism rings refract light that a mirror would lose, collapsing a bulky lens into a
thin one and gathering nearly all the source output into a horizontal beam. The single largest
improvement in the history of coastal lighting.

Orders run from first (largest, landfall) to sixth (smallest, harbour). Order is about focal
distance, not brightness: a first-order lens on a weak source is still a first-order lens.

### 2.3 Rotating assemblies

Mounting several lens panels on a rotating carriage gives a flash as each panel sweeps past.
Rotation period sets the character (§4). Early carriages ran on rollers; later ones floated
on a mercury bath, which is nearly frictionless and severely toxic.

## 3. Range

### 3.1 Nominal and actual

**Nominal range** is the distance the light would be visible in clear air. **Geographic range**
is set by the curvature of the Earth and depends on the height of the light and the observer's
height of eye. The useful range is whichever is smaller, and in poor visibility neither applies.

In haze, expect about **one third** of the nominal range.

*(Note: `../navigation/Navigation.md` §5.2 says one half. The documents disagree and the
disagreement is recorded in `../navigation/Study Plan.md` §2 rather than resolved.)*

### 3.2 Height and the horizon

Raising a light extends geographic range with the square root of height, so doubling the tower
height buys about forty per cent more range — an expensive way to gain distance, which is why
landfall lights are sited on headlands rather than built taller.

A light can also be **too high**: on a coast prone to low cloud, a lamp above the cloud base is
invisible from sea level while a lower one would show.

## 4. Character

A light is identified by its rhythm, not its brightness, and the rhythm is chosen so that
adjacent lights on the same coast cannot be mistaken for one another.

- **Fixed (F)** — continuous, no variation.
- **Flashing (Fl)** — light shorter than dark.
- **Occulting (Oc)** — dark shorter than light; the inverse of flashing.
- **Isophase (Iso)** — equal light and dark.
- **Group flashing (Fl(3))** — a repeating group of a stated number of flashes.
- **Alternating (Al)** — changes colour.

The period is the time for one complete cycle, and it is timed from the start of one flash to
the start of the next — not between flashes, which is the common error.

## 5. Sectors

A light may be screened to show different colours over different arcs: white over the safe
channel, red and green over the dangers either side. The boundary between sectors is not
perfectly sharp — expect a degree or two of uncertainty — so a sector boundary is a warning,
never a line to steer along.

## 6. What this map does not cover

Fog signals. Buoyage. Automation and remote monitoring.
