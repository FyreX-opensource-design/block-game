import json
import logging
import struct
import threading
import time
import zlib
from pathlib import Path
from flask import Flask, jsonify, request, Response

import numpy as np
from server.blocks import BLOCKS, HOTBAR_SLOTS, TABS
from server.items import ITEMS, ITEM_ATLAS, creative_items, stack_limit
from server.stacks import empty_stack, make_stack, normalize_stack, stack_count, stack_id
from client.settings import get_local_ips

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Block Game Companion</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; -webkit-user-select: none; }
  body {
    background-color: #121314;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    touch-action: manipulation;
    overflow-x: hidden;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  header {
    width: 100%;
    max-width: 680px;
    background: #1b1c1e;
    border-bottom: 2px solid #2a2c30;
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
  }
  .title { font-weight: bold; font-size: 16px; color: #4ade80; }
  .ptp-badge {
    background: #25282c;
    border: 1px solid #3c4046;
    border-radius: 4px;
    padding: 4px 8px;
    font-family: monospace;
    font-size: 11px;
    color: #38bdf8;
  }
  .container {
    width: 100%;
    max-width: 680px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }
  .panel {
    background: #202225;
    border: 2px solid #36393f;
    border-radius: 8px;
    padding: 14px;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  .panel-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-weight: 600;
    font-size: 15px;
    color: #f3f4f6;
  }
  .held-bar {
    width: 100%;
    background: #282a2e;
    border: 1px solid #38bdf8;
    border-radius: 6px;
    padding: 8px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .held-info {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .held-info img {
    width: 28px;
    height: 28px;
    image-rendering: pixelated;
  }
  .held-btn {
    padding: 6px 12px;
    background: #3b82f6;
    border: 1px solid #60a5fa;
    border-radius: 4px;
    color: white;
    font-size: 12px;
    font-weight: bold;
    cursor: pointer;
  }
  .held-btn-clear {
    background: #4b5563;
    border-color: #6b7280;
  }
  .slot-grid {
    display: grid;
    gap: 6px;
  }
  .grid-hotbar { grid-template-columns: repeat(9, 1fr); }
  .grid-storage { grid-template-columns: repeat(9, 1fr); }
  .grid-craft-2x2 { grid-template-columns: repeat(2, 54px); }
  .grid-craft-3x3 { grid-template-columns: repeat(3, 54px); }
  .slot {
    width: 54px;
    height: 54px;
    background: #151618;
    border: 2px solid #3a3d42;
    border-radius: 6px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    touch-action: none;
    transition: background 0.1s, border-color 0.1s;
  }
  @media (max-width: 540px) {
    .slot { width: 36px; height: 36px; }
    .grid-craft-2x2 { grid-template-columns: repeat(2, 42px); }
    .grid-craft-3x3 { grid-template-columns: repeat(3, 42px); }
    .slot-grid { gap: 4px; }
  }
  .slot:hover { background: #26292e; border-color: #5865f2; }
  .slot.selected { border-color: #facc15; box-shadow: 0 0 6px #facc1588; }
  .slot.drag-over { background: #32353b; border-color: #38bdf8; }
  .slot img {
    width: 80%;
    height: 80%;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
    pointer-events: none;
  }
  .slot .count {
    position: absolute;
    bottom: 2px;
    right: 4px;
    font-size: 13px;
    font-weight: bold;
    font-family: monospace;
    color: #ffffff;
    text-shadow: 1px 1px 0 #000, -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000;
    pointer-events: none;
  }
  .crafting-area {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 14px;
  }
  .arrow {
    font-size: 26px;
    color: #9ca3af;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
  }
  .furnace-area {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 14px;
  }
  .furnace-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }
  .furnace-fire {
    width: 24px;
    height: 24px;
    background: #333;
    border-radius: 4px;
    position: relative;
    overflow: hidden;
  }
  .furnace-fire-inner {
    position: absolute;
    bottom: 0;
    width: 100%;
    background: linear-gradient(to top, #ef4444, #f59e0b);
    transition: height 0.1s linear;
  }
  .cook-arrow-box {
    width: 36px;
    height: 18px;
    background: #2b2d31;
    border-radius: 3px;
    position: relative;
    overflow: hidden;
  }
  .cook-arrow-progress {
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    background: #10b981;
    transition: width 0.1s linear;
  }
  .controls-bar {
    width: 100%;
    display: flex;
    gap: 8px;
    margin-top: 10px;
  }
  .btn {
    flex: 1;
    padding: 10px 12px;
    background: #2b2d31;
    border: 1px solid #3f4248;
    color: #e5e7eb;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    text-align: center;
    transition: background 0.15s, border-color 0.15s;
  }
  .btn:active, .btn.active {
    background: #3b82f6;
    border-color: #60a5fa;
    color: white;
  }
  .btn.active-warn {
    background: #eab308;
    border-color: #fde047;
    color: #000;
  }
  .creative-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    width: 100%;
    margin-bottom: 12px;
  }
  .tab-btn {
    padding: 6px 10px;
    background: #1e1f22;
    border: 1px solid #33363b;
    border-radius: 4px;
    color: #9ca3af;
    font-size: 12px;
    cursor: pointer;
  }
  .tab-btn.active {
    background: #3b82f6;
    border-color: #60a5fa;
    color: #fff;
  }
  .creative-grid {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    gap: 6px;
    max-height: 220px;
    overflow-y: auto;
    width: 100%;
    padding-right: 4px;
    margin-bottom: 14px;
  }
  #held-cursor {
    position: fixed;
    pointer-events: none;
    z-index: 9999;
    width: 50px;
    height: 50px;
    display: none;
    align-items: center;
    justify-content: center;
    transform: translate(-50%, -50%);
  }
  #held-cursor img {
    width: 80%;
    height: 80%;
    image-rendering: pixelated;
  }
  #held-cursor .count {
    position: absolute;
    bottom: 2px;
    right: 4px;
    font-size: 13px;
    font-weight: bold;
    font-family: monospace;
    color: #ffffff;
    text-shadow: 1px 1px 0 #000, -1px -1px 0 #000;
  }
</style>
</head>
<body>

<header>
  <div>
    <div class="title">Block Game Companion</div>
    <div id="lan-ip" style="font-size:11px; color:#9ca3af; margin-top:2px;"></div>
  </div>
  <div id="ptp-status" class="ptp-badge">PTP: Syncing...</div>
</header>

<div class="container">
  <!-- When not in a world -->
  <div id="no-world-panel" class="panel" style="display:none; text-align:center; padding:36px 20px;">
    <div style="font-size:36px; margin-bottom:12px;">🌍</div>
    <div style="font-size:18px; font-weight:bold; color:#f3f4f6; margin-bottom:8px;">No Active World</div>
    <div style="font-size:13px; color:#9ca3af; max-width:380px; line-height:1.5; margin-bottom:18px;">
      Join a singleplayer world or connect to a multiplayer server in the game client to view and interact with your inventory.
    </div>
    <div id="no-world-status" class="ptp-badge" style="display:inline-block; font-size:12px; padding:6px 14px;">
      Status: In Main Menu
    </div>
  </div>

  <div id="container-panel" class="panel">
    <div class="panel-header">
      <span id="panel-title">Inventory</span>
      <span id="gamemode-tag" style="font-size:12px; color:#9ca3af; text-transform:capitalize;"></span>
    </div>

    <!-- Held item info bar -->
    <div id="held-bar" class="held-bar" style="display:none;">
      <div class="held-info">
        <span style="color:#9ca3af; font-size:13px;">Holding:</span>
        <img id="held-bar-img" src="" alt="" />
        <strong id="held-bar-name" style="font-size:14px; color:#fff;"></strong>
        <span id="held-bar-count" style="font-size:12px; color:#facc15; font-family:monospace;"></span>
      </div>
      <div style="display:flex; gap:6px;">
        <button class="held-btn" onclick="placeHeldInSelected()">Place in Hotbar</button>
        <button class="held-btn held-btn-clear" onclick="clearCursor()">✕ Put Away</button>
      </div>
    </div>

    <!-- Active container content rendered dynamically -->
    <div id="top-content"></div>

    <div class="panel-header" style="margin-top:10px; margin-bottom:8px;">
      <span>Storage</span>
    </div>
    <div id="storage-grid" class="slot-grid grid-storage"></div>

    <div class="panel-header" style="margin-top:12px; margin-bottom:8px;">
      <span>Hotbar</span>
    </div>
    <div id="hotbar-grid" class="slot-grid grid-hotbar"></div>

    <div class="controls-bar">
      <button id="btn-split" class="btn" onclick="toggleSplitMode()">Place 1 / Split: OFF</button>
      <button id="btn-quick" class="btn" onclick="toggleQuickMove()">Quick Move: OFF</button>
      <button id="btn-clear" class="btn" onclick="clearCursor()">Clear Cursor</button>
    </div>
  </div>
</div>

<div id="held-cursor">
  <img id="held-img" src="" alt="">
  <span id="held-count" class="count"></span>
</div>

<script>
  let state = null;
  let splitMode = false;
  let quickMove = false;
  let ptpOffset = 0.0;
  let ptpDelay = 0.0;
  let lastPtpSync = 0;

  // Pointer drag state
  let activePointerType = null;
  let isPointerDown = false;
  let dragStart = null;
  let isDragging = false;
  let longPressTimer = null;

  function ptpMasterNow() {
    return (performance.now() / 1000.0) - ptpOffset;
  }

  async function syncPtp() {
    try {
      const t1_client = performance.now() / 1000.0;
      const res1 = await fetch('/api/ptp/sync');
      const d1 = await res1.json();
      const t2_client = performance.now() / 1000.0;
      const t1_master = d1.t1;

      const t3_client = performance.now() / 1000.0;
      const res2 = await fetch('/api/ptp/delay', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ t3: t3_client })
      });
      const d2 = await res2.json();
      const t4_master = d2.t4;

      const delay = ((t4_master - t1_master) - (t3_client - t2_client)) / 2.0;
      const offset = ((t2_client - t1_master) - (t4_master - t3_client)) / 2.0;

      ptpOffset = offset;
      ptpDelay = delay;
      lastPtpSync = Date.now();

      const offsetMs = (offset * 1000).toFixed(1);
      const delayMs = (delay * 1000).toFixed(1);
      document.getElementById('ptp-status').textContent = `PTP: Master Synced (${offsetMs >= 0 ? '+' : ''}${offsetMs}ms, RTT: ${(delay*2000).toFixed(1)}ms)`;
    } catch (e) {
      document.getElementById('ptp-status').textContent = 'PTP: Connecting...';
    }
  }

  function toggleSplitMode() {
    splitMode = !splitMode;
    const btn = document.getElementById('btn-split');
    btn.textContent = `Place 1 / Split: ${splitMode ? 'ON' : 'OFF'}`;
    btn.className = `btn ${splitMode ? 'active-warn' : ''}`;
  }

  function toggleQuickMove() {
    quickMove = !quickMove;
    const btn = document.getElementById('btn-quick');
    btn.textContent = `Quick Move: ${quickMove ? 'ON' : 'OFF'}`;
    btn.className = `btn ${quickMove ? 'active' : ''}`;
  }

  async function clearCursor() {
    hideHeldCursor();
    await sendAction('clear', 'left', false);
  }

  function placeHeldInSelected() {
    if (state && state.selected !== undefined) {
      sendAction(`hotbar:${state.selected}`, 'left', false, { touch: true });
    }
  }

  async function sendAction(action, button = 'left', shift = false, extra = {}) {
    try {
      const payload = Object.assign({
        action: action,
        button: button,
        shift: shift || quickMove,
        timestamp: ptpMasterNow()
      }, extra);
      const res = await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        state = await res.json();
        render();
      }
    } catch (e) {
      console.error(e);
    }
  }

  function updateHeldCursor(x, y, iconId, count) {
    const cursor = document.getElementById('held-cursor');
    if (!cursor) return;
    if (iconId) {
      cursor.style.display = 'flex';
      cursor.style.left = `${x}px`;
      cursor.style.top = `${y}px`;
      document.getElementById('held-img').src = `/api/icon/${encodeURIComponent(iconId)}`;
      document.getElementById('held-count').textContent = count > 1 ? count : '';
    } else {
      cursor.style.display = 'none';
    }
  }

  function hideHeldCursor() {
    const cursor = document.getElementById('held-cursor');
    if (cursor) cursor.style.display = 'none';
  }

  function handleSlotTap(action, e) {
    const isTouch = activePointerType === 'touch';
    const btn = (e.button === 2 || splitMode) ? 'right' : 'left';

    if (action.startsWith('item:')) {
      const itemId = action.split(':')[1];
      if (state && state.held && state.held.id === itemId) {
        sendAction('clear', 'left', false);
        return;
      }
      if (isTouch) {
        // Direct equip full stack to active hotbar slot on tap
        sendAction(action, 'left', false, { touch: true, selected_hotbar: state ? state.selected : 0 });
        return;
      }
    }

    if (action.startsWith('hotbar:')) {
      const idx = parseInt(action.split(':')[1]);
      if (state && state.held && state.held.id) {
        sendAction(action, btn, quickMove, { touch: isTouch });
        return;
      }
      if (state && state.selected !== idx && !splitMode) {
        // Select this hotbar slot
        sendAction(`select_hotbar:${idx}`, 'left', false);
        return;
      }
    }

    sendAction(action, btn, quickMove || e.shiftKey, { touch: isTouch });
  }

  function setupSlotEvents(el, action, stack) {
    el.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      sendAction(action, 'right', quickMove);
    });

    el.addEventListener('pointerdown', (e) => {
      activePointerType = e.pointerType;
      isPointerDown = true;
      dragStart = { x: e.clientX, y: e.clientY, slotAction: action, stack: stack };
      isDragging = false;

      if (e.pointerType === 'touch') {
        longPressTimer = setTimeout(() => {
          longPressTimer = null;
          if (!isDragging) {
            sendAction(action, 'right', quickMove, { touch: true });
          }
        }, 380);
      }
    });

    el.addEventListener('click', (e) => {
      if (isDragging) return;
      handleSlotTap(action, e);
    });
  }

  window.addEventListener('pointermove', (e) => {
    activePointerType = e.pointerType;
    if (e.pointerType === 'mouse') {
      if (state && state.held && state.held.id) {
        updateHeldCursor(e.clientX, e.clientY, state.held.id, state.held.count);
      } else {
        hideHeldCursor();
      }
    } else if (isPointerDown && dragStart) {
      const dist = Math.hypot(e.clientX - dragStart.x, e.clientY - dragStart.y);
      if (dist > 8) {
        if (!isDragging) {
          isDragging = true;
          if (longPressTimer) {
            clearTimeout(longPressTimer);
            longPressTimer = null;
          }
        }
        const dragIcon = dragStart.stack?.id || (state && state.held ? state.held.id : '');
        const dragCount = dragStart.stack?.count || (state && state.held ? state.held.count : 1);
        if (dragIcon) {
          updateHeldCursor(e.clientX, e.clientY, dragIcon, dragCount);
        }
        const elem = document.elementFromPoint(e.clientX, e.clientY);
        const targetSlot = elem?.closest('.slot');
        document.querySelectorAll('.slot').forEach(s => s.classList.remove('drag-over'));
        if (targetSlot) targetSlot.classList.add('drag-over');
      }
    }
  });

  window.addEventListener('pointerup', (e) => {
    isPointerDown = false;
    document.querySelectorAll('.slot').forEach(s => s.classList.remove('drag-over'));

    if (e.pointerType === 'touch') {
      hideHeldCursor();
    }

    if (longPressTimer) {
      clearTimeout(longPressTimer);
      longPressTimer = null;
    }

    if (dragStart && isDragging) {
      const elem = document.elementFromPoint(e.clientX, e.clientY);
      const targetSlot = elem?.closest('.slot');
      if (targetSlot && targetSlot.dataset.action && targetSlot.dataset.action !== dragStart.slotAction) {
        const targetAction = targetSlot.dataset.action;
        if (dragStart.slotAction.startsWith('item:')) {
          sendAction('drop_creative', 'left', false, {
            item_id: dragStart.slotAction.split(':')[1],
            target: targetAction
          });
        } else {
          sendAction('move_slot', 'left', false, {
            from: dragStart.slotAction,
            to: targetAction
          });
        }
      }
    }
    dragStart = null;
    isDragging = false;
  });

  window.addEventListener('pointercancel', () => {
    isPointerDown = false;
    isDragging = false;
    dragStart = null;
    if (longPressTimer) {
      clearTimeout(longPressTimer);
      longPressTimer = null;
    }
    document.querySelectorAll('.slot').forEach(s => s.classList.remove('drag-over'));
    hideHeldCursor();
  });

  function createSlot(stack, action, isSelected = false) {
    const div = document.createElement('div');
    div.className = `slot ${isSelected ? 'selected' : ''}`;
    div.dataset.action = action;
    setupSlotEvents(div, action, stack);

    if (stack && stack.id) {
      const img = document.createElement('img');
      img.src = `/api/icon/${encodeURIComponent(stack.id)}`;
      img.alt = stack.name || stack.id;
      div.appendChild(img);

      if (stack.count > 1) {
        const countSpan = document.createElement('span');
        countSpan.className = 'count';
        countSpan.textContent = stack.count;
        div.appendChild(countSpan);
      }
    }
    return div;
  }

  function render() {
    if (!state) return;

    const lanElem = document.getElementById('lan-ip');
    if (lanElem) {
      const port = window.location.port || 5000;
      if (state.local_ips && state.local_ips.length > 0) {
        lanElem.textContent = 'Network: ' + state.local_ips.map(ip => `http://${ip}:${port}`).join('  |  ');
      } else {
        lanElem.textContent = `http://localhost:${port}`;
      }
    }

    const noWorldPanel = document.getElementById('no-world-panel');
    const containerPanel = document.getElementById('container-panel');
    if (!state.in_world) {
      if (noWorldPanel) noWorldPanel.style.display = 'flex';
      if (containerPanel) containerPanel.style.display = 'none';
      hideHeldCursor();
      const statusElem = document.getElementById('no-world-status');
      if (statusElem) {
        statusElem.textContent = `Status: ${state.game_status || 'In Main Menu'}`;
      }
      return;
    }
    if (noWorldPanel) noWorldPanel.style.display = 'none';
    if (containerPanel) containerPanel.style.display = 'flex';

    document.getElementById('gamemode-tag').textContent = state.gamemode || 'survival';
    const topContent = document.getElementById('top-content');
    topContent.innerHTML = '';

    const containerType = state.container_type || 'player';
    const panelTitle = document.getElementById('panel-title');

    // Update Held bar
    const heldBar = document.getElementById('held-bar');
    if (state.held && state.held.id) {
      heldBar.style.display = 'flex';
      document.getElementById('held-bar-img').src = `/api/icon/${encodeURIComponent(state.held.id)}`;
      document.getElementById('held-bar-name').textContent = state.held.name || state.held.id;
      document.getElementById('held-bar-count').textContent = state.held.count > 1 ? `x${state.held.count}` : '';
    } else {
      heldBar.style.display = 'none';
    }

    if (containerType === 'crafting_table') {
      panelTitle.textContent = 'Crafting Table';
      const area = document.createElement('div');
      area.className = 'crafting-area';

      const grid = document.createElement('div');
      grid.className = 'slot-grid grid-craft-3x3';
      (state.container_craft || []).forEach((stack, idx) => {
        grid.appendChild(createSlot(stack, `table_craft:${idx}`));
      });
      area.appendChild(grid);

      const arrow = document.createElement('div');
      arrow.className = 'arrow';
      arrow.innerHTML = '&#10140;';
      area.appendChild(arrow);

      const resSlot = createSlot(state.table_preview, 'table_result:out');
      area.appendChild(resSlot);
      topContent.appendChild(area);

    } else if (containerType === 'furnace' && state.furnace) {
      panelTitle.textContent = 'Furnace';
      const f = state.furnace;
      const area = document.createElement('div');
      area.className = 'furnace-area';

      const leftCol = document.createElement('div');
      leftCol.className = 'furnace-col';
      leftCol.appendChild(createSlot(f.input, 'furnace:input'));

      const fire = document.createElement('div');
      fire.className = 'furnace-fire';
      const fireInner = document.createElement('div');
      fireInner.className = 'furnace-fire-inner';
      fireInner.style.height = `${Math.min(100, Math.max(0, f.burn_progress * 100))}%`;
      fire.appendChild(fireInner);
      leftCol.appendChild(fire);

      leftCol.appendChild(createSlot(f.fuel, 'furnace:fuel'));
      area.appendChild(leftCol);

      const midCol = document.createElement('div');
      midCol.className = 'furnace-col';
      const cookBox = document.createElement('div');
      cookBox.className = 'cook-arrow-box';
      const cookProg = document.createElement('div');
      cookProg.className = 'cook-arrow-progress';
      cookProg.style.width = `${Math.min(100, Math.max(0, f.cook_progress * 100))}%`;
      cookBox.appendChild(cookProg);
      midCol.appendChild(cookBox);
      area.appendChild(midCol);

      const rightCol = document.createElement('div');
      rightCol.className = 'furnace-col';
      rightCol.appendChild(createSlot(f.output, 'furnace:output'));
      area.appendChild(rightCol);

      topContent.appendChild(area);

    } else if (state.gamemode === 'creative') {
      panelTitle.textContent = 'Creative Inventory';
      const tabsBar = document.createElement('div');
      tabsBar.className = 'creative-tabs';
      (state.creative_tabs || []).forEach(tab => {
        const btn = document.createElement('button');
        btn.className = `tab-btn ${tab.id === state.current_tab ? 'active' : ''}`;
        btn.textContent = tab.name;
        btn.onclick = () => sendAction(`tab:${tab.id}`);
        tabsBar.appendChild(btn);
      });
      topContent.appendChild(tabsBar);

      const grid = document.createElement('div');
      grid.className = 'creative-grid';
      (state.creative_items || []).forEach(id => {
        const slot = createSlot({ id: id, count: 1 }, `item:${id}`);
        grid.appendChild(slot);
      });
      topContent.appendChild(grid);

    } else {
      panelTitle.textContent = 'Crafting';
      const area = document.createElement('div');
      area.className = 'crafting-area';

      const grid = document.createElement('div');
      grid.className = 'slot-grid grid-craft-2x2';
      (state.craft || []).forEach((stack, idx) => {
        grid.appendChild(createSlot(stack, `craft:${idx}`));
      });
      area.appendChild(grid);

      const arrow = document.createElement('div');
      arrow.className = 'arrow';
      arrow.innerHTML = '&#10140;';
      area.appendChild(arrow);

      const resSlot = createSlot(state.craft_preview, 'result:out');
      area.appendChild(resSlot);
      topContent.appendChild(area);
    }

    const storageGrid = document.getElementById('storage-grid');
    storageGrid.innerHTML = '';
    (state.storage || []).forEach((stack, idx) => {
      storageGrid.appendChild(createSlot(stack, `store:${idx}`));
    });

    const hotbarGrid = document.getElementById('hotbar-grid');
    hotbarGrid.innerHTML = '';
    (state.hotbar || []).forEach((stack, idx) => {
      hotbarGrid.appendChild(createSlot(stack, `hotbar:${idx}`, idx === state.selected));
    });

    // If mouse user, show cursor
    if (activePointerType === 'mouse' && state.held && state.held.id) {
      document.getElementById('held-cursor').style.display = 'flex';
      document.getElementById('held-img').src = `/api/icon/${encodeURIComponent(state.held.id)}`;
      document.getElementById('held-count').textContent = state.held.count > 1 ? state.held.count : '';
    } else if (activePointerType === 'touch') {
      // Touch users never get a stuck frozen cursor when finger is up!
      hideHeldCursor();
    }
  }

  async function pollState() {
    try {
      const res = await fetch('/api/state');
      if (res.ok) {
        state = await res.json();
        render();
      }
    } catch (e) {
      console.warn("Poll failed", e);
    }
  }

  syncPtp();
  setInterval(syncPtp, 4000);
  pollState();
  setInterval(pollState, 150);
</script>
</body>
</html>
"""


def _encode_png(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = image.shape[:2]

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


class WebCompanion:
    def __init__(self, game, host="0.0.0.0", port=5000):
        self.game = game
        self.host = host
        self.port = int(port)
        self.app = Flask(__name__)
        self._server = None
        self._thread = None
        self._running = False
        self._icon_cache = {}
        self._setup_routes()

    def _get_item_icon_bytes(self, item_id):
        if item_id in self._icon_cache:
            return self._icon_cache[item_id]

        target_id = item_id
        if target_id not in ITEMS and target_id in BLOCKS:
            target_id = getattr(BLOCKS[target_id], "base_id", target_id)

        item = ITEMS.get(target_id)
        if item is None or not hasattr(item, "icon") or item.icon is None:
            # Fallback transparent 16x16
            empty = np.zeros((16, 16, 4), dtype=np.uint8)
            png = _encode_png(empty)
            self._icon_cache[item_id] = png
            return png

        W, H = ITEM_ATLAS.shape[1], ITEM_ATLAS.shape[0]
        x0 = int(round(item.icon[0] * W - 0.5))
        x1 = int(round(item.icon[2] * W + 0.5))
        y0 = int(round(item.icon[1] * H - 0.5))
        y1 = int(round(item.icon[3] * H + 0.5))
        tile = np.flipud(ITEM_ATLAS[y0:y1, x0:x1])
        png = _encode_png(tile)
        self._icon_cache[item_id] = png
        return png

    def _slot_to_dict(self, slot):
        if not slot:
            return {"id": "", "count": 0, "name": ""}
        sid = stack_id(slot)
        cnt = stack_count(slot)
        name = sid
        if sid in ITEMS:
            name = ITEMS[sid].name
        elif sid in BLOCKS:
            name = BLOCKS[sid].name
        return {"id": sid, "count": cnt, "name": name}

    def _get_state_dict(self):
        with getattr(self.game, "lock", threading.RLock()):
            in_world = getattr(self.game, "world", None) is not None and getattr(self.game, "mode", "") in ("play", "pause", "inventory")
            is_loading = getattr(self.game, "mode", "") == "loading"
            game_status = "In World" if in_world else ("Loading World..." if is_loading else "In Main Menu")

            inv = self.game.inventory
            container_type = getattr(inv, "container_type", "player")

            craft_slots = [self._slot_to_dict(s) for s in getattr(self.game, "craft", [])]
            craft_preview = self._slot_to_dict(getattr(inv, "preview", empty_stack()))

            table_craft = [self._slot_to_dict(s) for s in getattr(inv, "container_craft", [])]
            table_preview = self._slot_to_dict(getattr(inv, "table_preview", empty_stack()))

            furnace_data = None
            if container_type == "furnace" and inv.furnace is not None:
                f = inv.furnace
                b_dur = f.burn_duration if f.burn_duration > 0 else 1.0
                c_dur = f.cook_duration if f.cook_duration > 0 else 10.0
                furnace_data = {
                    "input": self._slot_to_dict(f.input),
                    "fuel": self._slot_to_dict(f.fuel),
                    "output": self._slot_to_dict(f.output),
                    "burn_time": float(f.burn_time),
                    "burn_duration": float(b_dur),
                    "burn_progress": float(max(0.0, min(1.0, f.burn_time / b_dur))),
                    "cook_time": float(f.cook_time),
                    "cook_duration": float(c_dur),
                    "cook_progress": float(max(0.0, min(1.0, f.cook_time / c_dur))),
                    "is_burning": f.burn_time > 0,
                }

            tabs = []
            for t in TABS:
                if self.game.gamemode == "creative" or t.screen == "player":
                    tabs.append({"id": t.id, "name": t.name, "screen": t.screen})

            c_items = []
            if self.game.gamemode == "creative":
                c_items = list(creative_items(inv.tab))

            held_id = getattr(inv, "held", "")
            held_name = ""
            if held_id in ITEMS:
                held_name = ITEMS[held_id].name
            elif held_id in BLOCKS:
                held_name = BLOCKS[held_id].name

            return {
                "in_world": in_world,
                "is_loading": is_loading,
                "game_status": game_status,
                "master_time": time.monotonic(),
                "local_ips": get_local_ips(),
                "gamemode": self.game.gamemode,
                "game_mode_screen": getattr(self.game, "mode", "play"),
                "container_type": container_type,
                "container_block": getattr(inv, "container_block", None),
                "selected": getattr(self.game, "selected", 0),
                "held": {
                    "id": held_id,
                    "count": getattr(inv, "held_count", 0),
                    "name": held_name,
                },
                "hotbar": [self._slot_to_dict(s) for s in getattr(self.game, "hotbar", [])],
                "storage": [self._slot_to_dict(s) for s in getattr(self.game, "storage", [])],
                "craft": craft_slots,
                "craft_preview": craft_preview,
                "container_craft": table_craft,
                "table_preview": table_preview,
                "furnace": furnace_data,
                "creative_tabs": tabs,
                "current_tab": inv.tab,
                "creative_items": c_items,
            }

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return Response(HTML_CONTENT, mimetype="text/html")

        @self.app.route("/api/state")
        def get_state():
            return jsonify(self._get_state_dict())

        @self.app.route("/api/ptp/sync")
        def ptp_sync():
            return jsonify({"t1": time.monotonic()})

        @self.app.route("/api/ptp/delay", methods=["POST"])
        def ptp_delay():
            t4 = time.monotonic()
            data = request.get_json(silent=True) or {}
            t3 = data.get("t3", 0.0)
            return jsonify({"t3": t3, "t4": t4})

        @self.app.route("/api/ptp", methods=["POST"])
        def ptp_compact():
            t_recv = time.monotonic()
            data = request.get_json(silent=True) or {}
            t1_slave = data.get("t1_slave", 0.0)
            return jsonify({
                "t1_slave": t1_slave,
                "t2_master": t_recv,
                "t3_master": time.monotonic(),
            })

        @self.app.route("/api/icon/<item_id>")
        def get_icon(item_id):
            png_bytes = self._get_item_icon_bytes(item_id)
            return Response(png_bytes, mimetype="image/png", headers={"Cache-Control": "public, max-age=86400"})

        @self.app.route("/api/action", methods=["POST"])
        def handle_action():
            data = request.get_json(silent=True) or {}
            action = data.get("action")
            button = data.get("button", "left")
            shift = bool(data.get("shift", False))

            with getattr(self.game, "lock", threading.RLock()):
                in_world = getattr(self.game, "world", None) is not None and getattr(self.game, "mode", "") in ("play", "pause", "inventory")
                if not in_world:
                    return jsonify(self._get_state_dict())

                prev_slots = self.game._capture_furnace_slots() if hasattr(self.game, "_capture_furnace_slots") else None

                if action == "clear":
                    self.game.inventory.held = ""
                    self.game.inventory.held_count = 0
                    self.game.inventory.held_infinite = False

                elif action and action.startswith("select_hotbar:"):
                    idx = int(action.split(":", 1)[1])
                    self.game.selected = max(0, min(HOTBAR_SLOTS - 1, idx))

                elif action == "drop_creative":
                    item_id = data.get("item_id")
                    target = data.get("target")
                    if item_id and target:
                        limit = stack_limit(item_id)
                        if target.startswith("hotbar:"):
                            h_idx = int(target.split(":", 1)[1])
                            self.game.hotbar[h_idx] = make_stack(item_id, limit)
                        elif target.startswith("store:"):
                            s_idx = int(target.split(":", 1)[1])
                            self.game.storage[s_idx] = make_stack(item_id, limit)
                        self.game.inventory.held = ""
                        self.game.inventory.held_count = 0
                        self.game.inventory.held_infinite = False

                elif action == "move_slot":
                    from_action = data.get("from")
                    to_action = data.get("to")
                    if from_action and to_action and from_action != to_action:
                        self.game.inventory.apply(
                            from_action,
                            "left",
                            self.game.hotbar,
                            self.game.selected,
                            self.game.storage,
                            self.game.craft,
                            gamemode=self.game.gamemode,
                            furnace=self.game.inventory.furnace,
                        )
                        self.game.inventory.apply(
                            to_action,
                            "left",
                            self.game.hotbar,
                            self.game.selected,
                            self.game.storage,
                            self.game.craft,
                            gamemode=self.game.gamemode,
                            furnace=self.game.inventory.furnace,
                        )

                elif action and action.startswith("item:"):
                    item_id = action.split(":", 1)[1]
                    if self.game.inventory.held == item_id:
                        # Tapping already selected item toggles/clears it
                        self.game.inventory.held = ""
                        self.game.inventory.held_count = 0
                        self.game.inventory.held_infinite = False
                    elif data.get("touch", False) and data.get("selected_hotbar") is not None:
                        # Direct tap equips full stack to selected hotbar slot on touch devices
                        sel_idx = int(data.get("selected_hotbar", 0))
                        if 0 <= sel_idx < HOTBAR_SLOTS:
                            limit = stack_limit(item_id)
                            self.game.hotbar[sel_idx] = make_stack(item_id, limit)
                            self.game.inventory.held = ""
                            self.game.inventory.held_count = 0
                            self.game.inventory.held_infinite = False
                    else:
                        picked = self.game.inventory.apply(
                            action,
                            button,
                            self.game.hotbar,
                            self.game.selected,
                            self.game.storage,
                            self.game.craft,
                            gamemode=self.game.gamemode,
                            furnace=self.game.inventory.furnace,
                            shift=shift,
                        )
                        if picked is not None:
                            self.game.selected = picked

                else:
                    was_infinite = getattr(self.game.inventory, "held_infinite", False)
                    held_before = self.game.inventory.held

                    picked = self.game.inventory.apply(
                        action,
                        button,
                        self.game.hotbar,
                        self.game.selected,
                        self.game.storage,
                        self.game.craft,
                        gamemode=self.game.gamemode,
                        furnace=self.game.inventory.furnace,
                        shift=shift,
                    )
                    if picked is not None:
                        self.game.selected = picked

                    # In web UI, if placing an infinite creative item into a slot, give a full stack and release it
                    if was_infinite and held_before and action and (action.startswith("hotbar:") or action.startswith("store:")):
                        limit = stack_limit(held_before)
                        if action.startswith("hotbar:"):
                            idx = int(action.split(":", 1)[1])
                            self.game.hotbar[idx] = make_stack(held_before, limit)
                        elif action.startswith("store:"):
                            idx = int(action.split(":", 1)[1])
                            self.game.storage[idx] = make_stack(held_before, limit)
                        self.game.inventory.held = ""
                        self.game.inventory.held_count = 0
                        self.game.inventory.held_infinite = False

                # Update previews
                if hasattr(self.game.inventory, "_preview"):
                    self.game.inventory.preview = self.game.inventory._preview(self.game.craft)
                if hasattr(self.game.inventory, "_table_preview"):
                    self.game.inventory.table_preview = self.game.inventory._table_preview(self.game.inventory.container_craft)

                if hasattr(self.game, "_sync_furnace_changes") and prev_slots is not None:
                    self.game._sync_furnace_changes(prev_slots)

            return jsonify(self._get_state_dict())

    def start(self):
        if self._running:
            return
        from werkzeug.serving import make_server

        try:
            self._server = make_server(self.host, self.port, self.app, threaded=True)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            self._running = True
        except Exception as e:
            print(f"[WebCompanion] Failed to start server on {self.host}:{self.port} - {e}")
            self._running = False

    def stop(self):
        if not self._running or self._server is None:
            return
        try:
            self._server.shutdown()
        except Exception:
            pass
        self._running = False
        self._server = None

    def is_running(self):
        return self._running
