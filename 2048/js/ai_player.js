/**
 * ai_player.js  –  DQN AI for 2048
 *
 * Loaded BEFORE application.js so we can wrap the GameManager constructor
 * and expose `window.gameManager` for later use.
 *
 * Requires:  onnxruntime-web (loaded via CDN in index.html)
 * Models at: ../onnx/{key}_agent.onnx  (serve project root with HTTP)
 */

// ─── 1. Wrap GameManager to expose instance ───────────────────────────────────
(function () {
  var _GM = window.GameManager;
  window.GameManager = function (size, InputManager, Actuator, StorageManager) {
    _GM.call(this, size, InputManager, Actuator, StorageManager);
    window.gameManager = this;
  };
  window.GameManager.prototype = _GM.prototype;
  window.GameManager.prototype.constructor = window.GameManager;
})();

// ─── 2. Point ONNX wasm workers to CDN ───────────────────────────────────────
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// ─── 3. AI Player ─────────────────────────────────────────────────────────────
var AI = (function () {
  'use strict';

  // Must match Python env: 0=up, 1=right, 2=down, 3=left
  var AGENTS = [
    { key: 'baseline', label: 'Baseline' },
    { key: 'partial',  label: 'Partial'  },
    { key: 'full',     label: 'Full'     },
    { key: 'best',     label: 'Best'     },
  ];

  var session   = null;
  var loadedKey = null;
  var timer     = null;
  var btnEl, agentEl, speedEl;

  // Grid → Float32Array[256] one-hot (mirrors Python _preprocess_state)
  // Python get_state: for x in 0..3, for y in 0..3 → cells[x][y]
  function encodeBoard() {
    var cells = window.gameManager.grid.cells;
    var state = new Float32Array(256);
    var i = 0;
    for (var x = 0; x < 4; x++) {
      for (var y = 0; y < 4; y++) {
        var tile = cells[x][y];
        var k    = tile ? Math.round(Math.log2(tile.value)) : 0;
        k = Math.min(Math.max(k, 0), 15);
        state[i * 16 + k] = 1.0;
        i++;
      }
    }
    return state;
  }

  // Mirrors Python env.get_legal_actions():
  // A direction is legal if any tile has an empty cell or a same-value tile
  // immediately in front of it (in the movement direction).
  // vectors: 0=up {x:0,y:-1}, 1=right {x:1,y:0}, 2=down {x:0,y:1}, 3=left {x:-1,y:0}
  var VECTORS = [{x:0,y:-1},{x:1,y:0},{x:0,y:1},{x:-1,y:0}];

  function getLegalActions() {
    var cells = window.gameManager.grid.cells;
    var legal = [];
    for (var dir = 0; dir < 4; dir++) {
      var v = VECTORS[dir];
      outer:
      for (var x = 0; x < 4; x++) {
        for (var y = 0; y < 4; y++) {
          var tile = cells[x][y];
          if (!tile) continue;
          var nx = x + v.x, ny = y + v.y;
          if (nx < 0 || nx >= 4 || ny < 0 || ny >= 4) continue;
          var next = cells[nx][ny];
          if (!next || next.value === tile.value) { legal.push(dir); break outer; }
        }
      }
    }
    return legal;
  }

  async function step() {
    if (!window.gameManager || window.gameManager.isGameTerminated()) {
      stop(); updateButton(); return;
    }

    var legal = getLegalActions();
    if (!legal.length) { stop(); updateButton(); return; }

    var tensor  = new ort.Tensor('float32', encodeBoard(), [1, 256]);
    var out     = await session.run({ state: tensor });
    var q       = Array.from(out['q_values'].data);

    // Mask illegal actions with -Infinity, then argmax
    for (var a = 0; a < 4; a++) {
      if (legal.indexOf(a) === -1) q[a] = -Infinity;
    }
    var action = q.indexOf(Math.max.apply(null, q));

    window.gameManager.inputManager.emit('move', action);
  }

  async function start() {
    var key     = agentEl.value;
    var delayMs = parseInt(speedEl.value, 10);

    if (key !== loadedKey) {
      btnEl.textContent = 'Loading…';
      btnEl.disabled = true;
      try {
        session   = await ort.InferenceSession.create('onnx/' + key + '_agent.onnx');
        loadedKey = key;
      } catch (e) {
        alert(
          'Cannot load model.\n\n' +
          'Serve the project root with HTTP, e.g.:\n' +
          '  python -m http.server 8000\n' +
          'then open: http://localhost:8000/2048/'
        );
        updateButton(); return;
      }
    }

    timer = setInterval(step, delayMs);
    updateButton();
  }

  function stop() {
    clearInterval(timer); timer = null;
  }

  function toggle() {
    if (timer) { stop(); updateButton(); }
    else       { start(); }
  }

  function updateButton() {
    btnEl.textContent = timer ? '⏹ Stop' : '▶ AI Play';
    btnEl.disabled    = false;
  }

  // ── Inject UI panel between .above-game and .game-container ──────────────
  function injectUI() {
    var panel = document.createElement('div');
    panel.id = 'ai-panel';
    panel.style.cssText = 'margin:10px 0;display:flex;align-items:center;gap:8px;flex-wrap:wrap';

    // Agent selector
    agentEl = document.createElement('select');
    agentEl.style.cssText = 'padding:6px 8px;border-radius:3px;border:2px solid #bbada0;background:#f9f6f2;font-size:14px;font-family:inherit;cursor:pointer';
    AGENTS.forEach(function (a) {
      var opt = document.createElement('option');
      opt.value = a.key;
      opt.textContent = a.label + ' AI';
      agentEl.appendChild(opt);
    });
    agentEl.addEventListener('change', function () { loadedKey = null; });

    // Speed label + slider
    var lbl = document.createElement('label');
    lbl.textContent = 'Speed:';
    lbl.style.cssText = 'font-size:13px;color:#776e65';

    speedEl = document.createElement('input');
    speedEl.type  = 'range';
    speedEl.min   = 50; speedEl.max = 800; speedEl.value = 250; speedEl.step = 50;
    speedEl.style.cssText = 'width:80px;cursor:pointer;vertical-align:middle';
    speedEl.title = 'Move delay (ms) – left = faster';
    speedEl.addEventListener('input', function () { if (timer) { stop(); start(); } });

    // Start / Stop button  (matches existing .restart-button style)
    btnEl = document.createElement('a');
    btnEl.textContent = '▶ AI Play';
    btnEl.className   = 'restart-button';
    btnEl.style.cssText = 'cursor:pointer;user-select:none';
    btnEl.addEventListener('click', toggle);

    panel.appendChild(agentEl);
    panel.appendChild(lbl);
    panel.appendChild(speedEl);
    panel.appendChild(btnEl);

    var ref = document.querySelector('.above-game');
    if (ref) ref.parentNode.insertBefore(panel, ref.nextSibling);
  }

  document.addEventListener('DOMContentLoaded', injectUI);

  return { toggle: toggle };
})();
