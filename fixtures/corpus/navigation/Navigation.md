# Navigation — knowledge map

> Entry point for position-finding at sea. This is a **map**, not a notebook: it holds
> framings and citations, while the catalogue of instruments lives in
> `Instruments Inventory.md` and the reading order in `Study Plan.md`.

## 1. The problem

A vessel needs two numbers, latitude and longitude, and the two are not equally hard.
Latitude follows from the altitude of a known body above the horizon and was routine by
the fifteenth century. Longitude requires knowing the time at a reference meridian at the
instant of observation, which is why it stayed unsolved for three hundred years.

## 2. Celestial methods

### 2.1 Meridian altitude

Observe a body at its highest point, when it crosses the observer's meridian. Latitude is
then derived from the measured altitude and the body's declination for that date. The method
needs no clock, which is its whole advantage.

Accuracy is limited by the horizon rather than the instrument: a sea horizon displaced by
haze introduces an error of several minutes of arc. See §5.1 on dip.

### 2.2 Lunar distance

Measure the angle between the Moon and a listed star, then compare against tabulated
predictions to recover reference time. Workable, and appallingly laborious — a single
reduction ran to four hours of arithmetic before the tables were simplified.

Superseded in practice by the marine chronometer (`Instruments Inventory.md`, Chronometer),
but it remained the fallback whenever a chronometer's rate was suspect.

### 2.3 Sight reduction

Modern practice pre-computes an assumed position, calculates the altitude a body *should*
show there, and plots the difference as an intercept. Three bodies give three lines, and
the triangle they enclose is the fix. A tight triangle is not proof of accuracy: three
sights sharing one systematic error close beautifully and sit in the wrong place.

## 3. Dead reckoning

Position carried forward from a known point using course, speed and elapsed time. Cheap,
always available, and wrong in a way that accumulates: an unrecorded current of one knot
displaces a vessel twenty-four miles a day.

Dead reckoning is therefore never the answer on its own. It is the estimate a celestial or
terrestrial fix corrects, and the rate at which the two diverge is itself information — a
persistent offset in one direction means a current worth charting.

## 4. Terrestrial methods

### 4.1 Bearings and transits

Two compass bearings on identified landmarks give a fix; three give a fix plus a check. A
**transit** — two fixed objects observed in line — is better than either, because it needs
no instrument and no correction. Harbour approaches are built around them.

### 4.2 Soundings

A line of soundings matched against charted depths gives position along a track, and in
fog it is sometimes the only method available. It fails where the seabed is flat, which is
most of a sandy coast.

## 5. Sources of error

### 5.1 Dip and refraction

Height of eye raises the visible horizon, and the correction, *dip*, grows with the square
root of that height. Refraction bends light downward near the horizon, making bodies appear
higher than they are; the correction is largest for low altitudes and unreliable below about
five degrees. Practice is to avoid sights below five degrees entirely.

### 5.2 Light visibility

The listed range of a light assumes clear air and a standard height of eye. In haze the
useful range is commonly **half** the listed figure, and a navigator who plans an approach
on the listed range alone will make his landfall later than intended.

*(Note: `Lighthouses.md` §3.1 gives one third rather than one half. The two documents
disagree, and the disagreement has not been resolved.)*

### 5.3 Compass deviation

The vessel's own iron deflects the compass, by an amount that varies with heading. The
deviation table is specific to one vessel and invalid after any structural change. Deviation
is not variation: variation is a property of the Earth and appears on the chart
(`../cartography/Cartography.md` §4.2).

## 6. What this map does not cover

Radio and satellite methods. Tidal stream prediction, which belongs with charts.
