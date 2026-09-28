# Projections Inventory

One line per projection, with what it preserves and where it is used. Reasoning is in
`Cartography.md` §1 and §2.

## 1. Conformal (angle-preserving)

- **Mercator** — preserves angle; straight line is a rhumb line; area distortion grows without limit toward the poles; the standard sea chart. See `Cartography.md` §2.
- **Transverse Mercator** — Mercator rotated onto a meridian; small distortion in a narrow north–south strip; basis of most national grids.
- **Stereographic** — preserves angle; circles map to circles; used for polar charts where Mercator fails outright.
- **Lambert conformal conic** — preserves angle along two standard parallels; the aeronautical chart, because a great circle is nearly straight on it.

## 2. Equal-area

- **Albers equal-area conic** — preserves area; used for thematic maps of mid-latitude countries; unsuitable for navigation.
- **Mollweide** — preserves area, whole world in an ellipse; badly distorted shapes at the edges.
- **Sinusoidal** — preserves area and scale along every parallel; the interrupted form reduces edge distortion at the cost of a discontinuous map.

## 3. Compromise

- **Robinson** — preserves nothing exactly; chosen to look least wrong overall; a world wall map, never a chart.
- **Plate carrée** — latitude and longitude plotted as a plain square grid; trivial to compute, distorts everything, common as a data storage format rather than a map.

## 4. Special-purpose

- **Gnomonic** — straight line is a great circle, i.e. the shortest path; used to plan a great-circle route which is then transferred to a Mercator chart as a series of rhumb legs.
- **Polar azimuthal equidistant** — distance from the centre point is true in every direction; used for range charts.

## 5. Not held

Historical projections of purely antiquarian interest. Projections for celestial charts.
