import { useEffect, useRef, useState } from "react";

declare global { interface Window { AMap?: any; _AMapSecurityConfig?: { securityJsCode: string } } }

export type MapPoint = { name: string; longitude: number; latitude: number };

let sdkLoading: Promise<void> | undefined;
function loadMapSdk(key: string) {
  if (window.AMap) return Promise.resolve();
  if (!sdkLoading) {
    sdkLoading = new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(key)}`;
      script.onload = () => window.AMap ? resolve() : reject(new Error("地图 SDK 未就绪"));
      script.onerror = () => { script.remove(); reject(new Error("地图加载失败，请检查网络和地图 Key")); };
      document.head.appendChild(script);
    }).catch(error => { sdkLoading = undefined; throw error; });
  }
  return sdkLoading;
}

export function MapPanel({ points }: { points: MapPoint[] }) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<any>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const key = import.meta.env.VITE_AMAP_JS_KEY;
  useEffect(() => {
    if (!key || !container.current) return;
    let cancelled = false;
    const security = import.meta.env.VITE_AMAP_JS_SECURITY_CODE;
    if (security) window._AMapSecurityConfig = { securityJsCode: security };
    void loadMapSdk(key).then(() => {
      if (cancelled || !container.current) return;
      map.current = new window.AMap.Map(container.current, { zoom: 11, viewMode: "2D" });
      setReady(true);
    }).catch(reason => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => {
      cancelled = true;
      map.current?.destroy();
      map.current = null;
    };
  }, [key]);

  useEffect(() => {
    if (!ready || !map.current || !window.AMap) return;
    const currentMap = map.current;
    const markers = points.map((point, index) => {
      const label = document.createElement("span");
      label.textContent = `${index + 1}. ${point.name}`;
      return new window.AMap.Marker({
        position: [point.longitude, point.latitude], title: point.name,
        // 高德 label.content 接收 HTML 字符串；textContent 先安全转义景点名。
        label: { content: label.outerHTML, direction: "top" },
      });
    });
    currentMap.add(markers);
    if (markers.length) currentMap.setFitView(markers, false, [60, 60, 60, 60]);
    return () => { if (map.current === currentMap) currentMap.remove(markers); };
  }, [points, ready]);

  const hint = !key ? "配置 VITE_AMAP_JS_KEY 后展示景点位置"
    : error ?? (!ready ? "地图加载中…" : !points.length ? "选择方案后展示景点；若仍为空，请重新生成包含坐标的计划。" : null);
  return <section className="panel map-panel">
    <div className="panel-title"><span>行程地图</span><small>{points.length ? `${points.length} 个景点 · ` : ""}GCJ-02</small></div>
    <div className="map-canvas">
      <div ref={container} style={{ width: "100%", height: "100%" }} />
      {hint && <div className="map-placeholder">{hint}</div>}
    </div>
  </section>;
}
