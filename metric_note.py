import numpy as np


def repeat_reward_note(notes, rest_ind):
    reward = reward_penalize_repeating(notes, rest_ind)
    reward = np.array(reward).reshape((len(notes), 1)).astype(np.float32)
    return reward


def penalize_parallel(self_pitches, self_durations, partner_pitches, partner_durations, rest_ind, penalize=-1):
    rewards = np.zeros(len(self_pitches))
    for ind, (self_p, self_d, partner_p, partner_d) in enumerate(zip(
            self_pitches, self_durations, partner_pitches, partner_durations)):
        if rest_ind in self_p[-2:]:
            continue
        self_offsets = np.concatenate([[0], np.cumsum(self_d)])
        partner_offsets = np.concatenate([[0], np.cumsum(partner_d)])
        vertical_offset = self_offsets[-2]
        partner_ind = sum(partner_offsets < vertical_offset) - 1
        if rest_ind not in partner_p[partner_ind-1:partner_ind+1] and self_p[-1] - partner_p[partner_ind] == self_p[-2] - partner_p[partner_ind-1]:
            diff = abs(self_p[-1] - partner_p[partner_ind])
            if diff % 12 in [0, 6, 7, 8] or (diff - 13) % 12 == 0:
                rewards[ind] += penalize
            rewards[ind] += penalize
    return rewards


def rules_reward_note(notes, durations, rest_ind):
    """
    return rewards: 'repeat-reward', 'autocorelate-reward', 'highlow-reward', 'motif-reward', 'repeat-motif-reward'
    """
    reward1 = reward_penalize_repeating(notes, rest_ind)
    reward1 = np.array(reward1).reshape((len(notes), 1)).astype(np.float32)

    reward3 = reward_penalize_autocorrelation(notes)
    reward3 = np.array(reward3).reshape((len(notes), 1)).astype(np.float32)

    reward4 = reward_high_low_unique(notes, rest_ind)
    reward4 = np.array(reward4).reshape((len(notes), 1)).astype(np.float32)

    reward5 = reward_motif(notes, durations, rest_ind)
    reward5 = np.array(reward5).reshape((len(notes), 1)).astype(np.float32)

    reward6 = reward_repeated_pattern(notes, durations)
    reward6 = np.array(reward6).reshape((len(notes), 1)).astype(np.float32)

    reward7, reward8 = reward_repeated_motif(notes, durations, rest_ind)
    reward7 = np.array(reward7).reshape((len(notes), 1)).astype(np.float32)
    reward8 = np.array(reward8).reshape((len(notes), 1)).astype(np.float32)

    return [reward1, reward3, reward4, reward5, reward6, reward7, reward8]


def reward_penalize_repeating(notes, rest_ind, reward_amount=-100.0):
    sum_reward = np.zeros(len(notes))

    for ind, note in enumerate(notes):

        num_repeated = 0
        contains_breaks = False
        action_note = note[-1]

        for i in range(len(note) - 2, -1, -1):
            if note[i] == action_note:
                num_repeated += 1
            elif note[i] == rest_ind:
                contains_breaks = True
            else:
                break

        if action_note == rest_ind and num_repeated > 1:
            sum_reward[ind] += reward_amount
        elif not contains_breaks and num_repeated > 4:
            sum_reward[ind] += reward_amount
        elif contains_breaks and num_repeated > 6:
            sum_reward[ind] += reward_amount

    return sum_reward


def reward_non_repeating(penalty, reward_amount=0.1):
    """Rewards the model for not playing the same note over and over.
        penalty = reward_penalize_repeating(pitches)
    """
    return (penalty >= 0) * reward_amount


def reward_penalize_autocorrelation(notes, threshold=0.15, reward_amount=-3.):
    lags = [1, 2, 3, 4]
    sum_reward = np.zeros(len(notes))
    max_correlate = np.zeros(len(notes))
    for lag in lags:
        coeffs = autocorrelate(notes, lag=lag)
        max_correlate = np.maximum(max_correlate, coeffs)
    sum_reward[max_correlate > threshold] = max_correlate[max_correlate > threshold] * reward_amount
    return sum_reward


def autocorrelate(signal, lag=1, crop_len=10):
    if signal.shape[1] < crop_len:
        return 0
    signal = signal[:, -crop_len:]
    x = signal - signal.mean(1, keepdims=True)
    c0 = np.var(x, 1)
    coeff = (x[:, lag:] * x[:, :-lag]).mean(1) / c0
    coeff[np.isnan(coeff)] = 1.
    return np.abs(coeff)


def reward_high_low_unique(notes, rest_ind, preview_len=20, reward_amount=1.0):
    if preview_len > 0:
        notes = notes[:, -preview_len:]
    sum_reward = np.zeros(len(notes))
    sum_reward[(notes == notes.min(1, keepdims=True)).sum(1) == 1] += reward_amount
    tmp_notes = notes.copy()
    tmp_notes[tmp_notes == rest_ind] = -1
    sum_reward[(tmp_notes == tmp_notes.max(1, keepdims=True)).sum(1) == 1] += reward_amount
    return sum_reward


def reward_motif(notes, durations, rest_ind, bar_length=2, reward_amount=3.0):
    offset = np.cumsum(durations[:, ::-1], 1)
    num_notes = (offset <= bar_length + 1e-8).sum(1)
    unique_notes = [[] if num == 0 else set(note[-num:]) for note, num in zip(notes, num_notes)]
    num_unique_notes = np.array([len([i for i in note if i != rest_ind]) for note in unique_notes])
    sum_reward = np.zeros(len(notes))
    sum_reward[num_unique_notes >= 3] = reward_amount + (num_unique_notes[num_unique_notes >= 3] - 3) * 0.3
    return sum_reward


def reward_repeated_pattern(notes, durations):
    sum_repeats = np.zeros(len(notes))
    for ind, (note, duration) in enumerate(zip(notes, durations)):
        for num_notes in range(3, 10):
            if len(note) < 2*num_notes:
                break
            pattern = note[-num_notes:]
            if np.all(note[-num_notes*2:-num_notes] == pattern) and np.all(duration[-num_notes*2:-num_notes] == pattern):
                sum_repeats[ind] += num_notes * 10
                break
    return -sum_repeats


def reward_repeated_motif(notes, durations, rest_ind, bar_length=2., reward_amount=4.0):
    sum_reward = np.zeros(len(notes))
    shift_reward = np.zeros(len(notes))
    for ind, (note, duration) in enumerate(zip(notes, durations)):
        offsets = np.cumsum(duration[::-1])
        num_notes = sum(offsets <= bar_length + 1e-8)
        if num_notes == 0:
            continue
        motif = note[-num_notes:]
        base_motif = note[-num_notes:] - note[-num_notes]
        actual_notes = [a for a in motif if a != rest_ind]
        num_unique_notes = len(set(actual_notes))
        if num_unique_notes >= 3:
            # Check if the motif is in the previous composition.
            i = 0
            while i <= len(note) - 2*num_notes:
                if np.all(note[i:i+num_notes] - note[i] == base_motif):
                    motif_complexity_bonus = max(num_notes - 3, 0)
                    shift_reward[ind] += reward_amount + motif_complexity_bonus
                    if np.all(note[i:i + num_notes] == motif):
                        sum_reward[ind] += reward_amount + motif_complexity_bonus
                    i += num_notes
                else:
                    i += 1
    return sum_reward, shift_reward


def calculate_gap(notes, rest_ind):
    average_gap = 0.
    max_gap = 0.
    for ind, note in enumerate(notes):
        this_max_gap = 0
        this_average_gap = 0
        count = 0
        pos2 = len(note) - 1
        while pos2 > 0:
            while pos2 > 0 and note[pos2] == rest_ind:
                pos2 -= 1
            pos1 = pos2 - 1
            while pos1 > 0 and note[pos1] == rest_ind:
                pos1 -= 1
            if pos1 < 0:
                continue
            gap = abs(note[pos2] - note[pos1])
            this_average_gap += gap
            this_max_gap = max(this_max_gap, gap)
            count += 1
            pos2 -= 1
        average_gap += this_average_gap / count
        max_gap += this_max_gap
    return average_gap / len(notes), max_gap / len(notes)

