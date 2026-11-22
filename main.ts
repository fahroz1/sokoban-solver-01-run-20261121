// Sokoban solver: decompose the level into rooms off a hallway, solve each
// room push-optimally with A* (minimum-assignment heuristic + deadlock
// pruning), then stitch the room solutions together with hallway walks.
//
// Deterministic by construction: fixed node-expansion budgets only, no wall
// clock, no Date, no unseeded randomness. All paths come from the
// MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR environment variables (or the
// /input and /output defaults), never from the process working directory.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions: 0 = up, 1 = down, 2 = left, 3 = right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const OPP = [1, 0, 3, 2];

// Fixed per-room node-expansion budgets (work, never time).
const ASTAR_CAP = 1500000;
const GREEDY_CAP = 1500000;

const IDS = ['i001', 'i002', 'i003', 'i004', 'i005'];

// ---------------------------------------------------------------- parsing

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  box: Uint8Array;
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goal = new Uint8Array(size);
  const box = new Uint8Array(size);
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') box[i] = 1;
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goal, box, player };
}

// ---------------------------------------------------------- decomposition

interface RoomSpec {
  cells: number[]; // global indices of the room floor, sorted
  doorGlobal: number; // the room cell adjacent to the hallway
  entryGlobal: number; // the hallway cell adjacent to the door
}

function decompose(level: Level): RoomSpec[] {
  const { w, h, wall, player } = level;
  const hallRow = Math.floor(player / w);
  const inHall = (i: number): boolean => Math.floor(i / w) === hallRow;
  const seen = new Uint8Array(w * h);
  const rooms: RoomSpec[] = [];
  for (let i = 0; i < w * h; i++) {
    if (wall[i] || inHall(i) || seen[i]) continue;
    const cells: number[] = [];
    const stack = [i];
    seen[i] = 1;
    while (stack.length > 0) {
      const c = stack.pop() as number;
      cells.push(c);
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d];
        const ny = cy + DY[d];
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const nb = ny * w + nx;
        if (wall[nb] || inHall(nb) || seen[nb]) continue;
        seen[nb] = 1;
        stack.push(nb);
      }
    }
    cells.sort((a, b) => a - b);
    // Door: room cells adjacent to a hallway cell (exactly one by construction).
    let doorGlobal = -1;
    let entryGlobal = -1;
    for (const c of cells) {
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d];
        const ny = cy + DY[d];
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const nb = ny * w + nx;
        if (!wall[nb] && inHall(nb)) {
          if (doorGlobal < 0) {
            doorGlobal = c;
            entryGlobal = nb;
          }
        }
      }
      if (doorGlobal >= 0) break;
    }
    if (doorGlobal < 0) continue; // a pocket with no door: not a room
    rooms.push({ cells, doorGlobal, entryGlobal });
  }
  // Fixed order: top to bottom, then left to right (by door cell).
  rooms.sort((a, b) => a.doorGlobal - b.doorGlobal);
  return rooms;
}

// ------------------------------------------------------------------- room

interface Room {
  n: number; // number of room floor cells (local 0..n-1); entry cell is local n
  k: number; // number of boxes (= number of goals)
  w: number;
  h: number;
  entryLocal: number;
  doorGlobal: number;
  entryGlobal: number;
  globalOf: number[]; // local -> global index
  cellLocal: Int32Array; // global index -> local cell, or -1 (not room/entry)
  boxNbr: number[][]; // [cell][dir] -> local room cell, or -1 (wall / entry)
  playNbr: number[][]; // [cell][dir] -> local cell incl. entry, or -1 (wall)
  isGoal: Uint8Array; // over local cells
  goals: number[]; // local goal cells, sorted
  startBoxes: number[]; // local box cells, sorted
  dist: Int32Array[]; // dist[g] = shortest box-path lengths from goals[g]
  goalPerms: number[][]; // all permutations of goal indices
  hCache: Map<number, number>;
}

function allPerms(k: number): number[][] {
  const out: number[][] = [];
  const used = new Array<boolean>(k).fill(false);
  const cur: number[] = [];
  (function rec(): void {
    if (cur.length === k) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  })();
  return out;
}

function buildRoom(level: Level, spec: RoomSpec): Room | null {
  const cells = spec.cells;
  const n = cells.length;
  if (n + 1 > 62) return null; // local indices must fit the 6-bit packing
  const localOf = new Map<number, number>();
  cells.forEach((g, i) => localOf.set(g, i));
  const entryLocal = n;
  localOf.set(spec.entryGlobal, entryLocal);
  const globalOf = new Array<number>(n + 1);
  cells.forEach((g, i) => {
    globalOf[i] = g;
  });
  globalOf[n] = spec.entryGlobal;
  const cellLocal = new Int32Array(level.w * level.h).fill(-1);
  cells.forEach((g, i) => {
    cellLocal[g] = i;
  });
  cellLocal[spec.entryGlobal] = entryLocal;
  const boxNbr: number[][] = [];
  const playNbr: number[][] = [];
  for (let i = 0; i <= n; i++) {
    const g = globalOf[i];
    const x = g % level.w;
    const y = Math.floor(g / level.w);
    const bn = [-1, -1, -1, -1];
    const pn = [-1, -1, -1, -1];
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d];
      const ny = y + DY[d];
      if (nx < 0 || ny < 0 || nx >= level.w || ny >= level.h) continue;
      const ng = ny * level.w + nx;
      const li = localOf.get(ng);
      if (li === undefined) continue;
      pn[d] = li;
      if (li < n) bn[d] = li;
    }
    boxNbr.push(bn);
    playNbr.push(pn);
  }
  const isGoal = new Uint8Array(n + 1);
  const goals: number[] = [];
  const startBoxes: number[] = [];
  for (let i = 0; i < n; i++) {
    const g = globalOf[i];
    if (level.goal[g]) {
      isGoal[i] = 1;
      goals.push(i);
    }
    if (level.box[g]) startBoxes.push(i);
  }
  goals.sort((a, b) => a - b);
  startBoxes.sort((a, b) => a - b);
  if (goals.length !== startBoxes.length) return null;
  const k = goals.length;
  // Shortest box-path lengths from each goal, over room cells only (boxes
  // never stand on the entry cell).
  const dist: Int32Array[] = goals.map((g0) => {
    const d = new Int32Array(n).fill(-1);
    d[g0] = 0;
    const q = [g0];
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi];
      for (let dir = 0; dir < 4; dir++) {
        const nb = boxNbr[c][dir];
        if (nb < 0 || d[nb] >= 0) continue;
        d[nb] = d[c] + 1;
        q.push(nb);
      }
    }
    return d;
  });
  return {
    n,
    k,
    w: level.w,
    h: level.h,
    entryLocal,
    doorGlobal: spec.doorGlobal,
    entryGlobal: spec.entryGlobal,
    globalOf,
    cellLocal,
    boxNbr,
    playNbr,
    isGoal,
    goals,
    startBoxes,
    dist,
    goalPerms: allPerms(k),
    hCache: new Map<number, number>(),
  };
}

// Pack sorted box cells into one number (6 bits each, base 64).
function packBoxes(boxes: number[]): number {
  let key = 0;
  for (let i = boxes.length - 1; i >= 0; i--) key = key * 64 + boxes[i];
  return key;
}

function stateKey(boxes: number[], player: number): number {
  return player + 64 * packBoxes(boxes);
}

// Minimum-assignment heuristic: the cheapest one-to-one matching of boxes
// to goals by shortest-path distance (ignoring boxes). Admissible and
// consistent: one push moves one box one step, so the matching cost moves
// by at most 1.
function heuristic(room: Room, boxes: number[]): number {
  const key = packBoxes(boxes);
  const cached = room.hCache.get(key);
  if (cached !== undefined) return cached;
  let best = Infinity;
  for (const p of room.goalPerms) {
    let s = 0;
    for (let i = 0; i < room.k; i++) {
      const d = room.dist[p[i]][boxes[i]];
      if (d < 0) {
        s = Infinity;
        break;
      }
      s += d;
    }
    if (s < best) best = s;
  }
  room.hCache.set(key, best);
  return best;
}

// ------------------------------------------------------- player reachability

// Flood fill from `from` over room cells + entry cell, avoiding box cells.
// Returns the set of reachable local cells.
function reachableSet(room: Room, boxSet: Uint8Array, from: number): Uint8Array {
  const seen = new Uint8Array(room.n + 1);
  const q = [from];
  seen[from] = 1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    for (let d = 0; d < 4; d++) {
      const nb = room.playNbr[c][d];
      if (nb < 0 || seen[nb] || boxSet[nb]) continue;
      seen[nb] = 1;
      q.push(nb);
    }
  }
  return seen;
}

// Canonical representative of the player's reachable region: its smallest
// local cell. All player cells in one region are equivalent.
function canonicalPlayer(room: Room, boxSet: Uint8Array, from: number): number {
  const seen = reachableSet(room, boxSet, from);
  for (let i = 0; i <= room.n; i++) if (seen[i]) return i;
  return from;
}

// ------------------------------------------------------------- deadlocking

// True when the box configuration can never reach all-goals. Only sound
// prunes: corner/freeze (a box that can move in neither axis), 2x2 blocks,
// and wall segments with no goal that the box cannot leave.
function isDead(room: Room, boxes: number[], boxSet: Uint8Array): boolean {
  const { n, boxNbr, playNbr, isGoal } = room;

  // A cell is wall-or-box when it is not walkable floor, or holds a box.
  // (The entry cell is floor and never holds a box, so it breaks a block.)
  const wallOrBox = (c: number): boolean => c < 0 || boxSet[c] === 1;
  // Local cell at global (x, y), or -1 for a wall / outside the room.
  // (Coordinates, not a neighbour chain: the cell diagonally past a wall
  // is a cell of its own and may well be free floor.)
  const locAt = (x: number, y: number): number => {
    if (x < 0 || y < 0 || x >= room.w || y >= room.h) return -1;
    return room.cellLocal[y * room.w + x];
  };

  for (const b of boxes) {
    if (isGoal[b]) continue;
    // 2x2 blocks containing b: b at each of the four corners.
    const bg = room.globalOf[b];
    const bx = bg % room.w;
    const by = Math.floor(bg / room.w);
    const squares: number[][] = [];
    for (const dx of [0, -1]) {
      for (const dy of [0, -1]) {
        const x0 = bx + dx;
        const y0 = by + dy;
        squares.push([locAt(x0, y0), locAt(x0 + 1, y0), locAt(x0, y0 + 1), locAt(x0 + 1, y0 + 1)]);
      }
    }
    for (const sq of squares) {
      if (sq.every(wallOrBox)) return true;
    }
    // Wall segments: a genuine wall in direction d pins the box to the run
    // of cells sharing that wall; if the run is walled off at both ends and
    // holds no goal, the box can never reach one. The entry cell is NOT a
    // wall here: the player can stand on it and push the box away from it,
    // so a box next to the door is never pinned by the hallway side.
    for (let d = 0; d < 4; d++) {
      if (playNbr[b][d] >= 0) continue;
      const a1 = d < 2 ? 2 : 0;
      const a2 = d < 2 ? 3 : 1;
      const run: number[] = [b];
      let c = b;
      while (true) {
        const nb = boxNbr[c][a1];
        if (nb < 0 || playNbr[nb][d] >= 0) break;
        run.push(nb);
        c = nb;
      }
      const loBounded = boxNbr[run[run.length - 1]][a1] < 0;
      c = b;
      while (true) {
        const nb = boxNbr[c][a2];
        if (nb < 0 || playNbr[nb][d] >= 0) break;
        run.push(nb);
        c = nb;
      }
      const hiBounded = boxNbr[run[run.length - 1]][a2] < 0;
      if (loBounded && hiBounded && !run.some((x) => isGoal[x] === 1)) return true;
    }
  }

  // Freeze deadlock: a box that cannot move along either axis, where a
  // neighbour counts as blocking only when it is a wall or a frozen box.
  // Iterate to a fixpoint; a frozen box off a goal is dead.
  const frozen = new Uint8Array(n + 1);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (frozen[b]) continue;
      const l = playNbr[b][2];
      const r = playNbr[b][3];
      const u = playNbr[b][0];
      const dn = playNbr[b][1];
      // standable: floor (room cell or entry) with no box, or a box that
      // can still move away. dest: a room cell with no box, or a box that
      // can still move away (boxes can never be pushed onto the entry cell).
      const stand = (cc: number): boolean => cc >= 0 && (!boxSet[cc] || !frozen[cc]);
      const dest = (cc: number): boolean => cc >= 0 && cc < n && (!boxSet[cc] || !frozen[cc]);
      const hMob = (stand(r) && dest(l)) || (stand(l) && dest(r));
      const vMob = (stand(dn) && dest(u)) || (stand(u) && dest(dn));
      if (!hMob && !vMob) {
        frozen[b] = 1;
        changed = true;
      }
    }
  }
  for (const b of boxes) if (frozen[b] && !isGoal[b]) return true;
  return false;
}

// ------------------------------------------------------------------- search

interface Push {
  box: number; // local cell of the box before the push
  dir: number;
}

interface SearchNode {
  boxes: number[];
  player: number; // canonical player cell
  g: number;
  h: number;
  f: number;
  parent: SearchNode | null;
  pushBox: number;
  pushDir: number;
  seq: number;
}

interface SearchResult {
  pushes: Push[];
  expansions: number;
  exitable: boolean;
}

// Small binary heap ordered by (f, h, seq) for deterministic tie-breaking.
class Heap {
  a: SearchNode[] = [];
  get size(): number {
    return this.a.length;
  }
  push(node: SearchNode): void {
    const a = this.a;
    a.push(node);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (cmp(a[p], a[i]) <= 0) break;
      const t = a[p];
      a[p] = a[i];
      a[i] = t;
      i = p;
    }
  }
  pop(): SearchNode {
    const a = this.a;
    const top = a[0];
    const last = a.pop() as SearchNode;
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && cmp(a[l], a[m]) < 0) m = l;
        if (r < a.length && cmp(a[r], a[m]) < 0) m = r;
        if (m === i) break;
        const t = a[m];
        a[m] = a[i];
        a[i] = t;
        i = m;
      }
    }
    return top;
  }
}

function cmp(x: SearchNode, y: SearchNode): number {
  if (x.f !== y.f) return x.f - y.f;
  if (x.h !== y.h) return x.h - y.h;
  return x.seq - y.seq;
}

function isGoalState(room: Room, boxes: number[]): boolean {
  for (const b of boxes) if (!room.isGoal[b]) return false;
  return true;
}

function reconstruct(node: SearchNode): Push[] {
  const out: Push[] = [];
  let cur: SearchNode | null = node;
  while (cur !== null && cur.parent !== null) {
    out.push({ box: cur.pushBox, dir: cur.pushDir });
    cur = cur.parent;
  }
  out.reverse();
  return out;
}

// A* (mode 'astar', f = g + h) or greedy best-first (mode 'greedy', f = h),
// over pushes, with canonical player positions and full deadlock pruning.
// When a goal state is popped, it is returned at once if the player can
// still walk out of the room through the door; otherwise it is kept as a
// fallback and the search continues for an exitable goal of the same or
// slightly higher push count.
function searchRoom(room: Room, mode: 'astar' | 'greedy', cap: number): SearchResult | null {
  const startBoxes = room.startBoxes.slice().sort((a, b) => a - b);
  const startSet = new Uint8Array(room.n + 1);
  for (const b of startBoxes) startSet[b] = 1;
  const startPlayer = canonicalPlayer(room, startSet, room.entryLocal);
  const startH = heuristic(room, startBoxes);
  if (startH === Infinity) return null;
  if (isDead(room, startBoxes, startSet)) return null;

  const heap = new Heap();
  const closed = new Map<number, number>(); // state key -> best g seen
  let seq = 0;
  const f0 = mode === 'astar' ? startH : startH;
  heap.push({ boxes: startBoxes, player: startPlayer, g: 0, h: startH, f: f0, parent: null, pushBox: -1, pushDir: -1, seq: seq++ });
  closed.set(stateKey(startBoxes, startPlayer), 0);

  let expansions = 0;
  let fallback: SearchResult | null = null;
  while (heap.size > 0) {
    const node = heap.pop();
    const key = stateKey(node.boxes, node.player);
    const bestG = closed.get(key);
    if (bestG !== undefined && bestG < node.g) continue; // stale entry
    if (isGoalState(room, node.boxes)) {
      // Can the player still walk out through the door?
      const boxSet = new Uint8Array(room.n + 1);
      for (const b of node.boxes) boxSet[b] = 1;
      const reach = reachableSet(room, boxSet, node.player);
      const exitable = reach[room.entryLocal] === 1;
      const res: SearchResult = { pushes: reconstruct(node), expansions, exitable };
      if (exitable) return res;
      if (fallback === null) fallback = res;
      continue; // do not expand goal states; look for an exitable one
    }
    if (++expansions > cap) break;
    const boxSet = new Uint8Array(room.n + 1);
    for (const b of node.boxes) boxSet[b] = 1;
    const reach = reachableSet(room, boxSet, node.player);
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const b = node.boxes[bi];
      for (let d = 0; d < 4; d++) {
        const dest = room.boxNbr[b][d];
        if (dest < 0 || boxSet[dest]) continue;
        const behind = room.playNbr[b][OPP[d]];
        if (behind < 0 || !reach[behind]) continue;
        const nb = node.boxes.slice();
        nb[bi] = dest;
        nb.sort((a, x) => a - x);
        const nbSet = new Uint8Array(room.n + 1);
        for (const x of nb) nbSet[x] = 1;
        const newPlayer = canonicalPlayer(room, nbSet, b);
        const nkey = stateKey(nb, newPlayer);
        const g2 = node.g + 1;
        const prev = closed.get(nkey);
        if (prev !== undefined && prev <= g2) continue;
        if (isDead(room, nb, nbSet)) continue;
        const h2 = heuristic(room, nb);
        if (h2 === Infinity) continue;
        closed.set(nkey, g2);
        heap.push({
          boxes: nb,
          player: newPlayer,
          g: g2,
          h: h2,
          f: mode === 'astar' ? g2 + h2 : h2,
          parent: node,
          pushBox: b,
          pushDir: d,
          seq: seq++,
        });
      }
    }
  }
  return fallback;
}

// ------------------------------------------------------------ replay + stitch

// Shortest walk (letters) from `from` to `to` over room cells + entry cell,
// avoiding the given local box cells. Null when unreachable. Deterministic:
// fixed neighbour order u, d, l, r and a FIFO queue.
function roomWalk(room: Room, boxes: number[], from: number, to: number): string | null {
  if (from === to) return '';
  const boxSet = new Uint8Array(room.n + 1);
  for (const b of boxes) boxSet[b] = 1;
  const prev = new Int16Array(room.n + 1).fill(-1);
  const prevDir = new Int8Array(room.n + 1).fill(-1);
  const q = [from];
  prev[from] = from;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    for (let d = 0; d < 4; d++) {
      const nb = room.playNbr[c][d];
      if (nb < 0 || prev[nb] >= 0 || boxSet[nb]) continue;
      prev[nb] = c;
      prevDir[nb] = d;
      if (nb === to) {
        const out: string[] = [];
        let cur = to;
        while (cur !== from) {
          out.push(LETTER[prevDir[cur]]);
          cur = prev[cur];
        }
        out.reverse();
        return out.join('');
      }
      q.push(nb);
    }
  }
  return null;
}

interface RoomReplay {
  moves: string; // entry cell -> solved, ending at the final player cell
  finalBoxes: number[]; // local box cells after the pushes
  finalPlayer: number; // local player cell after the pushes
  exitable: boolean;
  exitMoves: string; // final player cell -> entry cell ('' when not exitable)
}

// Turn a room's push list into player moves, starting at the entry cell.
function replayRoom(room: Room, pushes: Push[]): RoomReplay | null {
  const boxes = room.startBoxes.slice();
  let player = room.entryLocal;
  const out: string[] = [];
  for (const p of pushes) {
    const behind = room.playNbr[p.box][OPP[p.dir]];
    const walk = roomWalk(room, boxes, player, behind);
    if (walk === null) return null;
    out.push(walk);
    out.push(LETTER[p.dir]);
    const dest = room.boxNbr[p.box][p.dir];
    const bi = boxes.indexOf(p.box);
    if (bi < 0 || dest < 0) return null;
    boxes[bi] = dest;
    player = p.box;
  }
  const exitMoves = roomWalk(room, boxes, player, room.entryLocal);
  return {
    moves: out.join(''),
    finalBoxes: boxes,
    finalPlayer: player,
    exitable: exitMoves !== null,
    exitMoves: exitMoves ?? '',
  };
}

// Shortest walk (letters) on the whole level from `from` to `to`, over
// floor cells, avoiding the given global box cells. Null when unreachable.
function globalWalk(level: Level, boxGlobal: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const { w, h, wall } = level;
  const prev = new Int32Array(w * h).fill(-1);
  const prevDir = new Int8Array(w * h).fill(-1);
  const q = [from];
  prev[from] = from;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const nb = ny * w + nx;
      if (wall[nb] || prev[nb] >= 0 || boxGlobal[nb]) continue;
      prev[nb] = c;
      prevDir[nb] = d;
      if (nb === to) {
        const out: string[] = [];
        let cur = to;
        while (cur !== from) {
          out.push(LETTER[prevDir[cur]]);
          cur = prev[cur];
        }
        out.reverse();
        return out.join('');
      }
      q.push(nb);
    }
  }
  return null;
}

// Solve one level. Returns the move string, or '' when the level cannot be
// solved this way (an empty answer is still a valid, answered instance).
function solveLevel(level: Level): string {
  const specs = decompose(level);
  if (specs.length === 0) return '';
  interface Done {
    room: Room;
    pushes: Push[];
    replay: RoomReplay;
  }
  const done: Done[] = [];
  for (const spec of specs) {
    const room = buildRoom(level, spec);
    if (room === null) return '';
    if (room.k === 0) continue; // an empty pocket: nothing to solve there
    let res = searchRoom(room, 'astar', ASTAR_CAP);
    if (res === null) res = searchRoom(room, 'greedy', GREEDY_CAP);
    if (res === null) return '';
    const replay = replayRoom(room, res.pushes);
    if (replay === null) return '';
    done.push({ room, pushes: res.pushes, replay });
  }
  // Rooms the player can leave go first (in door order); a room whose
  // solved state traps the player goes last, where no exit is needed.
  done.sort((a, b) => {
    const ea = a.replay.exitable ? 0 : 1;
    const eb = b.replay.exitable ? 0 : 1;
    if (ea !== eb) return ea - eb;
    return a.room.doorGlobal - b.room.doorGlobal;
  });

  const boxGlobal = new Uint8Array(level.w * level.h);
  for (let i = 0; i < boxGlobal.length; i++) if (level.box[i]) boxGlobal[i] = 1;
  let player = level.player;
  const out: string[] = [];
  for (let ri = 0; ri < done.length; ri++) {
    const { room, replay } = done[ri];
    const last = ri === done.length - 1;
    const walk = globalWalk(level, boxGlobal, player, room.entryGlobal);
    if (walk === null) return '';
    out.push(walk);
    out.push(replay.moves);
    for (const b of room.startBoxes) boxGlobal[room.globalOf[b]] = 0;
    for (const b of replay.finalBoxes) boxGlobal[room.globalOf[b]] = 1;
    if (!last) {
      if (!replay.exitable) return ''; // a trapping room that is not last
      out.push(replay.exitMoves);
      player = room.entryGlobal;
    }
  }
  return out.join('');
}

// --------------------------------------------------------------------- main

function main(): void {
  for (const id of IDS) {
    let moves = '';
    try {
      const text = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
      moves = solveLevel(parseLevel(text));
    } catch {
      moves = '';
    }
    const dir = join(OUTPUT_DIR, id);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'solution.txt'), moves + '\n', 'utf8');
  }
}

main();
