"""
照度センサの ID・物理配置・正規化座標の共通定義。

センサは 6x6 のグリッドに 0.4 m 間隔で並んだマットに埋まっている。グリッド番号 "gx,gy" は
カメラから見て左手前が (0,0)、gx は右へ、gy は奥へ向かって増える（右奥が (5,5)）。

light.csv / environment.csv の列名はセンサ ID（ch 名, 例: ch0_pi2）で固定し、座標は
processed/<データ名>/sensor_coords.json にデータごとに書き出す。列名そのものを座標にしていた
旧形式では、マットをずらした収録で列名が変わり、学習済みモデルの列選択や複数データの結合が
壊れるため。

座標は 3 段で作る:
  1. グリッド番号 × GRID_PITCH_M でマット上のメートル座標にする
  2. マットの変位を当てる（rotate: 支点まわりに反時計回り +deg°、shift: (+dx, +dy) m）。
     どちらもカメラから見た向きで、上から見て x=右・y=奥の座標系（反時計回り = 正の角度）
  3. 正規化枠 FRAMES[frame] = (origin, span) で (m - origin) / span として [0, 1] にする

正規化枠は 2 種類ある（--sensor_frame で選ぶ。データごとの最大値では割らない）:
  legacy  : (0.0, 2.0)。グリッド (0,0) が 0、2 m が 1。マットを動かしていない従来データの既定で、
            旧形式の列名 "gx/5,gy/5" と同じ値になる
  special : (-0.6, 3.2)。2026-10-09 の special 収録（normal・shift・rotate など 12 件）用。
            マット中心 (1.0, 1.0) m を中心に x/y 同じ幅で、最も外に出る rotate の右手前の角
            （中心から 1.518 m）まで [0, 1] に収まる
同じ物理位置でも枠が違えば値が違うので、legacy のデータと special のデータを混ぜて学習・評価すると
座標の意味がずれる（dataset.py は混在を警告する）。枠からはみ出す配置は to_normalized がエラーにする。
"""
import math
import re

GRID_PITCH_M = 0.4
# 正規化枠 (origin_m, span_m)。x/y 共通
FRAMES = {"legacy": (0.0, 2.0), "special": (-0.6, 3.2)}
DEFAULT_FRAME = "legacy"
COORDS_FILENAME = "sensor_coords.json"

# ch名 → グリッド番号 "gx,gy" の固定マッピング（センサー物理配置から決定）
HEADER_MAP = {
    "ch0_pi2": "2,5", "ch1_pi2": "2,4", "ch2_pi2": "2,3",
    "ch0_pi1": "0,5", "ch1_pi1": "0,4", "ch2_pi1": "0,3",
    "ch4_pi1": "1,5", "ch5_pi1": "1,4", "ch6_pi1": "1,3",
    "ch0_pi3": "0,2", "ch1_pi3": "0,1", "ch2_pi3": "0,0",
    "ch3_pi3": "1,2", "ch4_pi3": "1,1", "ch5_pi3": "1,0",
    "ch0_pi4": "2,2", "ch1_pi4": "2,1", "ch2_pi4": "2,0",
    "ch0_pi5": "3,5", "ch1_pi5": "4,5", "ch2_pi5": "5,5",
    "ch3_pi5": "3,4", "ch4_pi5": "4,4", "ch5_pi5": "5,4",
    "ch0_pi6": "3,3", "ch1_pi6": "4,3", "ch2_pi6": "5,3",
    "ch0_pi7": "3,1", "ch1_pi7": "4,1", "ch2_pi7": "5,1",
    "ch3_pi7": "3,2", "ch4_pi7": "4,2", "ch5_pi7": "5,2",
    "ch0_pi8": "3,0", "ch1_pi8": "4,0", "ch2_pi8": "5,0",
}
GRID_TO_CH = {g: ch for ch, g in HEADER_MAP.items()}
GRID_MAX = max(int(v) for g in HEADER_MAP.values() for v in g.split(","))

_INT_PAIR = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")
_FLOAT_PAIR = re.compile(r"^\s*([0-9.]+)\s*,\s*([0-9.]+)\s*$")


def grid_of(ch: str):
    gx, gy = HEADER_MAP[ch].split(",")
    return int(gx), int(gy)


def to_ch_name(col: str):
    """
    列名を ch 名に直す。ch 名ならそのまま、旧形式なら対応する ch 名、どちらでもなければ None。

    旧形式は 2 通りある:
      - 整数グリッド番号 "2,5"（古い raw と、setup_data.py の旧版が書いた light_origin.csv）
      - それを GRID_MAX で割った正規化値 "0.4,1.0"（旧版 preprocess_real.py が書いた light.csv）
    """
    if col in HEADER_MAP:
        return col
    m = _INT_PAIR.match(col)
    if m:
        return GRID_TO_CH.get(f"{int(m.group(1))},{int(m.group(2))}")
    m = _FLOAT_PAIR.match(col)
    if m:
        gx, gy = (float(v) * GRID_MAX for v in m.groups())
        if abs(gx - round(gx)) < 1e-6 and abs(gy - round(gy)) < 1e-6:
            return GRID_TO_CH.get(f"{round(gx)},{round(gy)}")
    return None


def rename_to_ch(columns, log_prefix="[sensor]"):
    """
    先頭（timestamp）以外の列名を ch 名に直した新しいリストと、書き換えた列数を返す。
    直せない列名があれば ValueError（未知のセンサを黙って通すと座標が引けない）。
    """
    out = [columns[0]]
    unknown = []
    n_renamed = 0
    for col in columns[1:]:
        ch = to_ch_name(col)
        if ch is None:
            unknown.append(col)
            out.append(col)
        else:
            out.append(ch)
            n_renamed += ch != col
    if unknown:
        raise ValueError(f"{log_prefix} センサ ID に変換できない列名があります: {unknown}")
    if len(set(out)) != len(out):
        raise ValueError(f"{log_prefix} 変換後の列名が重複しています: {out}")
    return out, n_renamed


def parse_shift(value):
    """--sensor_shift の JSON '[dx, dy]'（メートル）を (dx, dy) にする。None は (0, 0)。"""
    if value is None:
        return (0.0, 0.0)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"sensor_shift は [dx, dy]（メートル）で指定してください: {value!r}")


def parse_rot(value):
    """
    --sensor_rot の JSON '{"deg": 15, "pivot": [5, 5]}' を (deg, (pgx, pgy)) にする。None は回転なし。
    pivot はグリッド番号（支点のセンサ）。
    """
    if value is None:
        return 0.0, None
    if isinstance(value, dict) and "deg" in value and "pivot" in value:
        p = value["pivot"]
        if isinstance(p, str):
            p = p.split(",")
        if len(p) == 2:
            return float(value["deg"]), (float(p[0]), float(p[1]))
    raise ValueError(f'sensor_rot は {{"deg": 15, "pivot": [5, 5]}} の形で指定してください: {value!r}')


def positions_m(chs, shift=(0.0, 0.0), rot_deg=0.0, pivot_grid=None):
    """
    各センサのメートル座標 {ch: (x, y)} を返す。回転（支点まわり、反時計回り正）→ 平行移動の順。
    """
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    if pivot_grid is None:
        px = py = 0.0
    else:
        px, py = pivot_grid[0] * GRID_PITCH_M, pivot_grid[1] * GRID_PITCH_M
    out = {}
    for ch in chs:
        gx, gy = grid_of(ch)
        x, y = gx * GRID_PITCH_M - px, gy * GRID_PITCH_M - py
        x, y = c * x - s * y + px, s * x + c * y + py
        out[ch] = (x + shift[0], y + shift[1])
    return out


def to_normalized(pos_m, frame=DEFAULT_FRAME, log_prefix="[sensor]"):
    """メートル座標を正規化枠 FRAMES[frame] で [0, 1] に正規化する。枠外があれば ValueError。"""
    origin, span = FRAMES[frame]
    out = {}
    outside = []
    for ch, (x, y) in pos_m.items():
        u = (x - origin) / span
        v = (y - origin) / span
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            outside.append(f"{ch}=({x:.3f}, {y:.3f}) m")
        out[ch] = (u, v)
    if outside:
        raise ValueError(
            f"{log_prefix} 正規化枠 {frame} [{origin}, {origin + span}] m の外に出るセンサがあります: "
            f"{outside}。マットを動かしたデータは --sensor_frame special を使うこと")
    return out

