# Cartography — knowledge map

> How charts are made and what they distort. The per-projection catalogue is
> `Projections Inventory.md`; navigation practice is `../navigation/Navigation.md`.

## 1. The impossibility

A sphere cannot be flattened without distortion. Every projection therefore chooses what to
preserve and what to sacrifice, and no projection preserves shape, area, distance and
direction at once. Choosing a chart means choosing which error you are willing to carry.

## 2. What a sea chart must preserve

For navigation the answer is **angle**. A navigator plots a course as a straight line and
steers a constant compass heading; a projection where that straight line is not a constant
heading is useless at sea regardless of its other virtues.

This is why the Mercator projection, whose area distortion is notorious, remains the sea
chart: a straight line on it is a rhumb line, a track of constant bearing. The price is that
Greenland looks the size of Africa, which matters to nobody steering a vessel.

## 3. Survey methods

### 3.1 Triangulation

Measure one baseline precisely, then build a network of triangles outward by measuring angles
only. Angles are far easier to measure accurately than distances, which is the whole reason
the method exists. Error accumulates with distance from the baseline, so long chains are tied
back to a second measured base wherever possible.

### 3.2 Running survey

Fix the vessel's position repeatedly while under way, recording depths and bearings to
features ashore. Fast, and much less accurate than triangulation — it inherits every error in
the vessel's own position. Most of a remote coast is charted this way, which is worth
remembering when reading a chart of one.

### 3.3 Sounding density

A chart shows the soundings taken, not the seabed. A shoal narrower than the spacing between
sounding lines is simply absent. Early surveys ran lines a mile apart; a rock is smaller than
that. This is why a chart's survey date and scale matter more than its neatness.

## 4. Reading a chart

### 4.1 Datum

Charted depths are measured from a datum, normally the lowest tide reasonably expected.
Actual depth is charted depth plus the height of tide, so charted depth is a worst case —
except where the datum was set from a short tidal record, in which case it is not.

### 4.2 Variation

The angle between true and magnetic north, printed on the chart inside a compass rose, with
the year of measurement and an annual rate of change. It must be updated for the current
year before use. Variation is a property of the Earth; **deviation** is a property of the
vessel and is not on the chart (`../navigation/Navigation.md` §5.3).

### 4.3 Symbols and abbreviations

The symbol set is conventional and large. The important habit is checking the chart's own
legend rather than assuming, because conventions differ between issuing authorities and
between editions of the same chart.

## 5. What this map does not cover

Land topography. Satellite geodesy. Tidal stream atlases, which are a separate publication.
