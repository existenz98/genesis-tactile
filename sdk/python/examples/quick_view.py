
import cv2, time, argparse, numpy as np

from tacto6d import Tacto6DClient, ClientConfig


def flow_to_color_bgr(vy, vx, max_flow=None):
    mag = np.sqrt(vy**2 + vx**2)
    ang = np.arctan2(vy, vx)
    if max_flow is None:
        max_flow = max(1e-6, np.percentile(mag, 95.0))
    H = (ang + np.pi) / (2*np.pi)
    S = np.ones_like(H, dtype=np.float32)
    V = np.clip(mag / max_flow, 0, 1)
    hsv = np.stack([H*179.0, S*255.0, V*255.0], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def draw_quiver(img, vy, vx, stride=16, scale=10.0, minlen=0.2):
    H, W = vy.shape
    out = img.copy()
    for y in range(stride//2, H, stride):
        for x in range(stride//2, W, stride):
            dx = vx[y, x]; dy = vy[y, x]
            if (dx*dx + dy*dy) < (minlen*minlen): continue
            x1 = int(round(x + scale*dx)); y1 = int(round(y + scale*dy))
            cv2.arrowedLine(out, (x, y), (x1, y1), (0,0,0), 1, tipLength=0.3)
    return out

def heatmap(p):
    vmin = float(np.percentile(p, 2)); vmax = float(np.percentile(p, 98)); 
    if vmax <= vmin + 1e-6: vmax = vmin + 1.0
    pn = np.clip((p - vmin)/(vmax - vmin), 0, 1)
    cm = cv2.applyColorMap((pn*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cm



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notify", default="ipc:///tmp/tacto6d.frame")
    ap.add_argument("--ctrl",   default="ipc:///tmp/tacto6d.ctrl")
    args = ap.parse_args()

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("flow",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("force",  cv2.WINDOW_NORMAL)

    tacto6d = Tacto6DClient(ClientConfig(notify_ep=args.notify, ctrl_ep=args.ctrl))

    tacto6d.start()

    try:
        while True:
            #print(f"[quick_view] >>> tacto6d.latest_frame()")
            fr = tacto6d.latest_frame(copy=True)
            #print(f"[quick_view] <<< tacto6d.latest_frame(), frame={fr}")

            if fr is not None:
                #print(f"[quick_view] received frame={fr}")

                # raw camera image
                cam = fr["camera"]
                cv2.imshow("camera", cam)

                # flow x,y
                vy, vx = fr["vy"], fr["vx"]
                flow_bgr = flow_to_color_bgr(vy, vx, max_flow=None)
                flow_q = draw_quiver(flow_bgr, vy, vx, stride=16, scale=8.0)
                cv2.imshow("flow", flow_q)

                # 3D force map
                p, tx, ty = fr["p"], fr["tx"], fr["ty"]
                hm = heatmap(p)
                # upscale + draw shear on coarse grid centers
                Hc, Wc = p.shape
                vis = cv2.resize(hm, (Wc*16, Hc*16), interpolation=cv2.INTER_NEAREST)
                Ys, Xs = np.mgrid[0:Hc, 0:Wc]; Ys = Ys*16+8; Xs = Xs*16+8
                vis = draw_quiver(vis, ty, tx, stride=1, scale=6.0, minlen=0.0)
                cv2.imshow("force", vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('1'): tacto6d.set_algo(1)
            if k == ord('2'): tacto6d.set_algo(2)
            if k == ord('3'): tacto6d.set_algo(3)
            if k == ord('q'): break

    finally:
        tacto6d.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
