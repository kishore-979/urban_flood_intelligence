# PHASE 0 — Dataset and Project Feasibility Study

## 1. Project Context

This project is intended to match the problem idea of urban flood risk prediction and safe-route planning. The required system includes:

- rainfall and weather data,
- terrain and elevation data,
- flood or inundation information,
- road-network data,
- an ML classifier,
- first-order logic / rule reasoning,
- forward chaining,
- backward chaining,
- an intelligent agent,
- A* route planning,
- a Streamlit-based prototype.

The key requirement is not to invent data or pretend that incomplete sources are available.

---

## 2. SIH Problem Verification

### STATUS: DATASET / OFFICIAL PROBLEM DESCRIPTION NOT VERIFIED

I did not find a reliable, public, official SIH 26085 project description during the current web verification. The problem number appears to be undocumented in the primary public material available from this environment, so I cannot honestly claim a verified SIH wording or exact official challenge text.

### Why this matters

- The project scope is conceptually valid, but the exact SIH source description is not confirmed.
- This means the project should be framed as a valid academic prototype aligned with the given problem statement, not as a formally verified official SIH dataset release.

### Best honest interpretation

The project should proceed as: a semester-scale urban flood-risk prediction and intelligent route-planning prototype for a selected region with real public data.

---

## 3. Feasibility of the Core Idea

### Overall assessment

The project is feasible as an academic prototype, but only if we limit the scope to a selected city or region and use real public data sources.

### Recommended geographic scope

I recommend a city/region with strong public data access and known flood risk. The strongest candidate is:

- Houston, Texas, USA (or the greater Harris County area)

### Why Houston is a strong choice

- NOAA weather station data is publicly available and directly accessible.
- USGS 3DEP elevation and terrain data is publicly available.
- OSM road network data is publicly available.
- Harris County flood-control authorities maintain official local flood information.
- FEMA flood maps are available for the area.
- This combination allows a realistic ML target and a genuine route-planning environment.

---

## 4. Verified Data Sources

### A. Rainfall / Weather Data

Dataset: NOAA Daily Summaries / NCEI weather station data

- Official source: https://www.ncei.noaa.gov/
- Access method: HTTP JSON dataset endpoint with station ID and date range
- File format: JSON (also CSV possible via export options)
- Geographic coverage: weather-station level, applicable to Houston area stations
- Time coverage: daily records over many years
- License / restrictions: public government data, typically non-restrictive for academic use
- Labels: no direct flood label, but the weather variables can be used as predictive features
- ML support: yes, for rainfall, temperature, humidity, wind, and daily anomaly features
- Join strategy: join by date and station or location centroid / nearest station
- Static or dynamic: dynamic time series
- Limitations: weather station data is not full street-level flood data; it is a proxy for local precipitation conditions

### Verification evidence

I verified that this endpoint is live and returns actual JSON data:

- https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00012918&startDate=2023-01-01&endDate=2023-01-05&format=json

This returned a valid list of daily weather records containing variables such as PRCP (precipitation), TMAX, TMIN, and related fields.

---

### B. Terrain / Elevation Data

Dataset: USGS 3D Elevation Program (3DEP) DEM

- Official source: https://apps.nationalmap.gov/
- Access method: National Map / ArcGIS REST service or downloadable DEM tiles
- File format: GeoTIFF / raster tiles / online raster service
- Geographic coverage: national-level US coverage; usable for city-level Houston region
- Time coverage: DEM baseline, typically static terrain data
- License / restrictions: public US federal data; generally usable for research and academic use
- Labels: no flood label, but supports elevation, slope, drainage, and terrain features
- ML support: yes, as terrain covariates
- Join strategy: spatial join to grid cells or road segments via longitude/latitude and raster extraction
- Static or dynamic: static
- Limitations: terrain is static and not itself a flood label

### Verification evidence

I verified that this service is live and returns valid JSON metadata:

- https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer?f=json

The service responded with currentVersion and valid metadata, confirming it exists and is publicly accessible.

---

### C. Road Network Data

Dataset: OpenStreetMap regional export (Texas)

- Official source: https://download.geofabrik.de/
- Access method: direct regional `.osm.pbf` or `.osm.bz2` download from Geofabrik
- File format: `.osm.pbf` or equivalent OSM extract
- Geographic coverage: Texas, with local Houston subset available through filtering
- Time coverage: snapshot data (not live traffic data)
- License / restrictions: OSM data license (OpenStreetMap Foundation license terms)
- Labels: no direct flood label, but attributes and geometry for roads are available
- ML support: yes, as road graph and segment metadata
- Join strategy: route network graph built from OSM geometries and local flood-risk scores mapped onto the nearest road segments
- Static or dynamic: static snapshot
- Limitations: road conditions are not real-time; they are only topology/geometry snapshot data

### Verification evidence

I verified that the data source is live:

- https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf

This URL returned HTTP 200 with application/octet-stream, confirming the file is available for download.

---

### D. Flood Information / Flood Zones

Dataset: FEMA National Flood Hazard Layer (NFHL) and/or local county flood-control data

- Official sources:
  - https://www.fema.gov/
  - https://www.hcfcd.org/
- Access method: FEMA GIS download or county flood-control dashboards/data portals
- File format: shapefile, GeoJSON, or GIS layers
- Geographic coverage: local floodplain and flood-risk mapping for the selected region
- Time coverage: static flood mapping plus local historical flood observations
- License / restrictions: public government data; use must follow agency terms
- Labels: can provide flood or floodplain labels, but not necessarily daily/road-level labels
- ML support: yes, as a flood-risk feature or target proxy
- Join strategy: spatial join between points or grid cells and flood zones
- Static or dynamic: mostly static flood mapping, with local historical flood observations for contextual labeling
- Limitations: not a real-time flood sensor system; may not be daily event labels at the road-segment level

### Verification evidence

The official FEMA and Harris County flood-control sites are live and reachable. I did not verify the exact downloadable file bundle in this environment, so the download method should be treated as official but needing a deliberate manual or scripted fetch step.

---

## 5. Data Gaps and Constraints

### DATASET NOT FOUND / NOT VERIFIED

The exact SIH 26085 official problem statement is not publicly verified in this environment.

This is not a failure of the project idea but a limitation of the requirement source. That limitation must be acknowledged honestly.

### Missing or difficult-to-obtain elements

- real-time street-level flood sensor data,
- ground-truth flooded-road labels at each segment,
- daily flood event labels for every road segment,
- fully labeled citywide flood dataset with road-level outcomes.

### Why the missing data matters

A true supervised road-level flood-risk model would ideally need labels showing which roads flooded in which conditions. That level of ground truth is rarely available publicly for a whole city in a ready-to-use form.

### Best legitimate alternative

Use a derived target that is clearly documented, such as:

- flood risk at zone level,
- road segment risk derived from flood-zone proximity + rainfall intensity + elevation + local drainage context,
- binary class based on a thresholded storm event or floodplain exposure score.

This is scientifically defensible if we explain the derivation and keep it transparent.

---

## 6. Recommended ML Target

The most defensible target is a binary or ordinal flood-risk classification at a spatial unit (cell or area) rather than a false claim of direct street-level flood prediction.

### Recommended target

- Label 1 = high flood-risk area / road segment under a specified rainfall and terrain condition
- Label 0 = lower-risk area / road segment

### How to justify it

The data available through NOAA, USGS, OSM, and flood-zone sources supports area-based risk estimation, not guaranteed road-level ground truth for actual flooding.

---

## 7. Project Feasibility Verdict

## FEASIBILITY STATUS

### YELLOW — Feasible with limitations

This project is feasible as an academic prototype, provided the scope is limited to a selected region and the target is framed realistically.

### Reason

The project can genuinely include:

- rainfall data,
- terrain data,
- flood zone information,
- road graph data,
- ML risk prediction,
- rule-based reasoning,
- A* route planning,
- a Streamlit demonstration.

The project cannot honestly claim full real-time flood forecasting, exact road-level flood prediction, or precise hydrological simulation without additional infrastructure and more detailed local observations.

---

## 8. Recommended Architecture

### Data layer

- NOAA weather station data for rainfall and meteorological features
- USGS DEM for elevation, slope, and terrain features
- OSM road graph for roads and intersections
- FEMA/HCFCD flood data as flood-risk background information

### Analysis layer

- clean and merge by date and location
- compute rainfall anomalies and extreme-event indicators
- derive slope, elevation, flood exposure features
- create a zone/road segment risk dataset
- train a Random Forest classifier

### Reasoning layer

- represent flood risk facts using predicates such as HighRainfall, LowElevation, HighFloodRisk, UnsafeRoad
- implement FOL and rule-based reasoning
- use forward chaining and backward chaining

### Planning layer

- environment model with nodes and edges
- A* search with flood-risk penalties in edge weight
- route comparison between shortest path and flood-aware route

### Interface layer

- Streamlit dashboard
- map visualization
- risk prediction and reasoning explanation
- route comparison panels

---

## 9. Assumptions to Make Explicitly

## ASSUMPTION 1

### What is assumed?

The selected region will be a city or county-level area with public weather, terrain, and road data available.

### Why it is necessary

This is necessary because the project must be real and reproducible.

### Evidence

Live public endpoints for NOAA, USGS, and OSM were verified in this environment.

### Effect on the project

This defines the project as a geographic prototype rather than a nation-scale forecasting system.

### Can it be removed later?

Yes, but only if we add more detailed flood and road data.

---

## 10. Recommended Next Step

Proceed to Phase 1: environment setup and data acquisition for the selected region, using the verified sources above.

### Next phase target

- create project structure,
- install Python environment,
- download NOAA, USGS, OSM data,
- prepare preprocessing notebook and project skeleton,
- define the exact stale/flood-risk target at a city-zone level.

---

## 11. Final Recommendation

The project should not be framed as a national, real-time flood warning system. It should be framed as:

A reproducible academic prototype for urban flood-risk prediction and A*-based safe route planning in a selected city, using public weather, terrain, flood-zone, and road-network data.

This is the most scientifically honest and technically feasible interpretation of the requirement.
