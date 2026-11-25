"""Real contiguous replay only: native horizon preferred, shorter cold-start fallback.

L frames train L-1 transitions. Never use L=1 or padding/cross-reset data.
The configured WM horizon remains unchanged. Every call retries that horizon.
"""

def sample_batches(buffer, horizon, batch_size, num_batches):
    assert horizon >= 2 and num_batches >= 1
    for length in range(horizon, 1, -1):
        try:
            first = buffer.sample(length, batch_size)
        except RuntimeError as exc:
            if str(exc) != f'no valid sequences L={length} n={buffer.num_transitions}':
                raise
            continue
        def batches():
            yield first
            for _ in range(num_batches - 1):
                yield buffer.sample(length, batch_size)
        return length, batches()
    raise RuntimeError('WM bootstrap has no real within-episode transition (needs >=2 frames)')


def install(learning):
    def wm_update(self):
        cfg = self.wm.cfg
        length, data = sample_batches(self.buffer, cfg.horizon, cfg.batch_size, cfg.num_batches)
        logs = self.wm.update(data)  # Native learner/optimizer, no synthetic targets.
        self.count['wm_calls'] += 1
        self.count['wm_optimizer_batches'] += cfg.num_batches
        key = 'wm_short_window_calls' if length < cfg.horizon else 'wm_native_window_calls'
        self.count[key] = self.count.get(key, 0) + 1
        key = f'wm_real_window_L{length}_batches'
        self.count[key] = self.count.get(key, 0) + cfg.num_batches
        return {**{k:float(v.detach()) for k,v in logs.items()},
                'wm_configured_window_frames': cfg.horizon,
                'wm_actual_window_frames': length,
                'wm_actual_scored_transitions': length - 1,
                'wm_short_window_bootstrap': length < cfg.horizon,
                'wm_short_window_calls': self.count.get('wm_short_window_calls', 0)}
    learning.Learner.wm_update = wm_update
