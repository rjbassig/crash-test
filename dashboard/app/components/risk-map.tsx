"use client";

import "maplibre-gl/dist/maplibre-gl.css";
import * as maplibregl from "maplibre-gl";
import type { MapGeoJSONFeature } from "maplibre-gl";
import proj4 from "proj4";
import { feature } from "topojson-client";
import { useEffect, useRef, useState } from "react";

// MapLibre v6 requires an explicit worker URL when bundled by Next.js.
maplibregl.setWorkerUrl(new URL("maplibre-gl/dist/maplibre-gl-worker.mjs", import.meta.url).toString());

// SDPD's published beat topology is in California State Plane Zone 6 (US feet),
// while MapLibre expects WGS84 longitude/latitude coordinates.
proj4.defs("EPSG:2230", "+proj=lcc +lat_0=32.1666666666667 +lon_0=-116.25 +lat_1=32.7833333333333 +lat_2=33.8833333333333 +x_0=2000000 +y_0=500000 +datum=NAD83 +units=us-ft +no_defs +type=crs");

function toWgs84(value: unknown): unknown {
  if (!Array.isArray(value)) return value;
  if (typeof value[0] === "number" && typeof value[1] === "number") {
    return proj4("EPSG:2230", "WGS84", [value[0], value[1]]);
  }
  return value.map(toWgs84);
}

function reprojectGeometry(geometry: GeoJSON.Geometry): void {
  if ("coordinates" in geometry) {
    geometry.coordinates = toWgs84(geometry.coordinates) as never;
    return;
  }
  geometry.geometries.forEach(reprojectGeometry);
}

export type Metric = { police_beat: string; predicted_mean: number; observed_mean: number; relative_risk: number; risk_rank: number; occurrence_probability: number };
type RiskView = "relative_risk" | "predicted_mean" | "observed_mean" | "residual";
type Props = { api: string; onSelect: (beat: string) => void; selectedBeat: string | null; view: RiskView };

export function RiskMap({ api, onSelect, selectedBeat, view }: Props) {
  const node = useRef<HTMLDivElement>(null); const mapRef = useRef<maplibregl.Map | null>(null); const [message, setMessage] = useState("Loading official SDPD beat boundaries…");
  useEffect(() => { let map: maplibregl.Map | null = null; let cancelled = false;
    fetch(`${api}/api/geography/beats`).then(async response => { if (!response.ok) throw new Error(`Map data unavailable (${response.status})`); return response.json(); }).then(data => {
      if (cancelled || !node.current) return;
      const geo = feature(data.topology, data.topology.objects.data) as unknown as GeoJSON.FeatureCollection;
      geo.features.forEach(entry => { if (entry.geometry) reprojectGeometry(entry.geometry); });
      const metrics = new Map<string, Metric>(data.metrics.map((entry: Metric) => [String(entry.police_beat), entry]));
      geo.features.forEach(entry => { const key = String(entry.properties?.beat ?? ""); const metric = metrics.get(key); entry.properties = { ...entry.properties, ...metric, beat: key, residual: metric ? metric.observed_mean - metric.predicted_mean : 0 }; });
      map = new maplibregl.Map({ container: node.current, style: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#101412" } }] }, center: [-117.14, 32.77], zoom: 9.6 }); mapRef.current = map;
      map.on("load", () => { if (!map) return; map.addSource("beats", { type: "geojson", data: geo }); map.addLayer({ id: "beat-fill", type: "fill", source: "beats", paint: { "fill-color": ["interpolate", ["linear"], ["coalesce", ["get", view], 0], 0, "#132237", 1, "#2368ae", 2.5, "#9ed6ff"], "fill-opacity": .82 } }); map.addLayer({ id: "beat-line", type: "line", source: "beats", paint: { "line-color": "#7eb8e5", "line-width": .55, "line-opacity": .55 } }); map.addLayer({ id: "beat-selected", type: "line", source: "beats", filter: ["==", ["get", "beat"], selectedBeat ?? ""], paint: { "line-color": "#f2f7fb", "line-width": 2.3 } }); map.on("mouseenter", "beat-fill", () => { map?.getCanvas().style.setProperty("cursor", "pointer"); }); map.on("mouseleave", "beat-fill", () => { map?.getCanvas().style.setProperty("cursor", ""); }); map.on("click", "beat-fill", event => { const item = event.features?.[0] as MapGeoJSONFeature | undefined; const beat = String(item?.properties?.beat ?? ""); if (!beat || !metrics.has(beat)) return; onSelect(beat); new maplibregl.Popup({ closeButton: false, offset: 8 }).setLngLat(event.lngLat).setHTML(`<strong>BEAT ${beat}</strong><br/>Observed weekly reports: ${Number(item?.properties?.observed_mean).toFixed(2)}<br/>Predicted weekly reports: ${Number(item?.properties?.predicted_mean).toFixed(2)}<br/>Relative risk: ${Number(item?.properties?.relative_risk).toFixed(2)}×`).addTo(map!); }); setMessage(""); });
    }).catch(error => setMessage(error.message));
    return () => { cancelled = true; map?.remove(); mapRef.current = null; };
  }, [api, onSelect, selectedBeat, view]);
  return <div className="map-frame"><div ref={node} className="risk-map" aria-label="Interactive map of aggregate police-beat collision risk" />{message && <div className="map-message">{message}</div>}<p>Official SDPD law-beat boundaries · aggregate predicted count, not point-level crash probability.</p></div>;
}
