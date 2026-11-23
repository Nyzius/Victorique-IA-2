(function () {
  'use strict';

  const STATES = new Set(['acordada', 'dormindo', 'ouvindo', 'pensando', 'falando', 'erro']);
  const EXPRESSIONS = new Set([
    'neutral', 'skeptical', 'annoyed', 'concerned', 'happy',
    'surprised', 'embarrassed', 'focused', 'sleepy', 'deduction'
  ]);
  const GESTURES = new Set(['page-turn', 'pipe', 'deduction', 'tea', 'hair-adjust']);
  const GESTURE_MS = {
    'page-turn': 1700,
    pipe: 3200,
    deduction: 2100,
    tea: 2600,
    'hair-adjust': 1900,
  };
  const IDLE_GESTURES = ['page-turn', 'pipe', 'tea', 'hair-adjust'];

  function manifestIds(items, fallback) {
    if (!Array.isArray(items) || !items.length) return new Set(fallback);
    const ids = items.map(item => typeof item === 'string' ? item : item?.id).filter(Boolean);
    return ids.length ? new Set(ids) : new Set(fallback);
  }

  class VictoriqueAvatarEngine {
    constructor(stage, manifest = {}) {
      this.stage = stage;
      this.states = manifestIds(manifest.states, STATES);
      this.expressions = manifestIds(manifest.expressions, EXPRESSIONS);
      this.gestures = manifestIds(manifest.gestures, GESTURES);
      this.gestureMs = { ...GESTURE_MS };
      if (Array.isArray(manifest.gestures)) {
        manifest.gestures.forEach(item => {
          if (item?.id && Number.isFinite(item.durationMs)) this.gestureMs[item.id] = item.durationMs;
        });
      }
      this.idleGestures = IDLE_GESTURES.filter(value => this.gestures.has(value));
      this.state = this.states.has(manifest.defaultState) ? manifest.defaultState : 'acordada';
      this.expression = this.expressions.has(manifest.defaultExpression)
        ? manifest.defaultExpression : 'neutral';
      this.gesture = '';
      this.lastGesture = '';
      this.revision = 0;
      this.motion = window.matchMedia
        ? window.matchMedia('(prefers-reduced-motion: reduce)')
        : { matches: false, addEventListener() {} };
      this.onMotionChange = () => {
        if (this.motion.matches) this.setGesture('');
        this.scheduleBlink();
        this.scheduleIdle();
      };
      this.motion.addEventListener('change', this.onMotionChange);
      this.stage.addEventListener('pointerdown', () => this.interact());
      this.stage.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          this.interact();
        }
      });
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) this.pause();
        else this.resume();
      });
      this.render();
      this.resume();
    }

    setState(value) {
      if (!this.states.has(value) || value === this.state) return false;
      this.state = value;
      if (value === 'dormindo' || value === 'erro') this.setGesture('');
      this.render();
      this.scheduleBlink();
      this.scheduleIdle();
      return true;
    }

    setExpression(value, holdMs = 22000) {
      if (!this.expressions.has(value)) return false;
      this.expression = value;
      this.revision += 1;
      const revision = this.revision;
      clearTimeout(this.expressionTimer);
      if (value !== 'neutral' && Number.isFinite(holdMs) && holdMs > 0) {
        this.expressionTimer = setTimeout(() => {
          if (this.revision !== revision) return;
          this.expression = 'neutral';
          this.render();
        }, holdMs);
      }
      this.render();
      return true;
    }

    setGesture(value, durationMs) {
      clearTimeout(this.gestureTimer);
      if (!value) {
        this.gesture = '';
        this.render();
        return true;
      }
      if (!this.gestures.has(value) || this.motion.matches || this.state === 'dormindo') return false;
      this.gesture = value;
      this.lastGesture = value;
      this.render();
      const duration = Number.isFinite(durationMs) ? durationMs : this.gestureMs[value];
      this.gestureTimer = setTimeout(() => {
        if (this.gesture !== value) return;
        this.gesture = '';
        this.render();
        this.scheduleIdle();
      }, duration);
      return true;
    }

    handleControl(value) {
      if (!value || value.type !== 'avatar') return false;
      let accepted = false;
      if (value.expression) accepted = this.setExpression(value.expression) || accepted;
      if (value.gesture) accepted = this.setGesture(value.gesture) || accepted;
      return accepted;
    }

    attend() {
      this.scheduleIdle(14000);
      if (this.state === 'dormindo') this.setState('acordada');
    }

    interact() {
      if (this.state === 'dormindo') {
        this.setState('acordada');
        this.setExpression('surprised', 1800);
        return;
      }
      const next = this.expression === 'embarrassed' ? 'annoyed' : 'embarrassed';
      this.setExpression(next, 2800);
      this.setGesture('hair-adjust', 1800);
      this.stage.classList.remove('is-tapped');
      void this.stage.offsetWidth;
      this.stage.classList.add('is-tapped');
    }

    blink() {
      if (this.motion.matches || document.hidden || this.state === 'dormindo') return;
      this.stage.classList.add('is-blinking');
      clearTimeout(this.blinkEndTimer);
      this.blinkEndTimer = setTimeout(() => this.stage.classList.remove('is-blinking'), 170);
    }

    scheduleBlink(delay) {
      clearTimeout(this.blinkTimer);
      if (this.motion.matches || document.hidden || this.state === 'dormindo') return;
      const wait = Number.isFinite(delay) ? delay : 3200 + Math.random() * 4100;
      this.blinkTimer = setTimeout(() => {
        this.blink();
        this.scheduleBlink();
      }, wait);
    }

    scheduleIdle(delay) {
      clearTimeout(this.idleTimer);
      if (
        this.motion.matches || document.hidden || this.state !== 'acordada' || this.gesture
      ) return;
      const wait = Number.isFinite(delay) ? delay : 15000 + Math.random() * 10000;
      this.idleTimer = setTimeout(() => {
        const choices = this.idleGestures.filter((value) => value !== this.lastGesture);
        if (!choices.length) return;
        this.setGesture(choices[Math.floor(Math.random() * choices.length)]);
      }, wait);
    }

    pause() {
      clearTimeout(this.blinkTimer);
      clearTimeout(this.idleTimer);
      this.stage.classList.remove('is-blinking');
    }

    resume() {
      if (document.hidden) return;
      this.scheduleBlink(1800 + Math.random() * 1800);
      this.scheduleIdle();
    }

    render() {
      this.stage.dataset.state = this.state;
      this.stage.dataset.expression = this.expression;
      this.stage.dataset.gesture = this.gesture;
      this.stage.setAttribute(
        'aria-label',
        `Retrato procedural de Victorique: ${this.state}, expressão ${this.expression}`
      );
    }
  }

  window.VictoriqueAvatarEngine = VictoriqueAvatarEngine;
  window.VICTORIQUE_AVATAR = Object.freeze({
    states: [...STATES], expressions: [...EXPRESSIONS], gestures: [...GESTURES]
  });
})();
