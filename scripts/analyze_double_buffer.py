def parse_time_ms(t):
    if 's' in t and 'ms' not in t:
        return float(t.replace('s', '')) * 1000
    elif 'ms' in t:
        return float(t.replace('ms', ''))
    return 0

# Double buffer timeline
data = """
973.48ms  5.1010ms GEMM
978.62ms  4.6817ms GEMM
982.39ms  6.4489ms NCCL
983.33ms  5.0682ms GEMM
988.93ms  6.2335ms NCCL
995.23ms  6.3967ms NCCL
1.00198s  4.3402ms GEMM
1.00633s  4.6023ms GEMM
1.01044s  6.5463ms NCCL
1.01097s  5.0147ms GEMM
1.01602s  4.5211ms GEMM
1.01971s  6.1437ms NCCL
"""

print("Double Buffer Timeline Analysis:")
print("="*70)

lines = [l.strip() for l in data.strip().split('\n') if l.strip()]
events = []

for line in lines:
    parts = line.split()
    start = parse_time_ms(parts[0])
    duration = parse_time_ms(parts[1])
    ktype = parts[2]
    end = start + duration
    events.append({'start': start, 'end': end, 'duration': duration, 'type': ktype})

# Check for overlaps
overlaps = []
for i in range(len(events)):
    for j in range(i+1, len(events)):
        e1, e2 = events[i], events[j]
        # Check if they overlap
        if e1['start'] < e2['end'] and e2['start'] < e1['end']:
            overlap_start = max(e1['start'], e2['start'])
            overlap_end = min(e1['end'], e2['end'])
            overlap_duration = overlap_end - overlap_start
            if overlap_duration > 0.1:  # More than 0.1ms overlap
                overlaps.append({
                    'e1': e1,
                    'e2': e2,
                    'overlap_ms': overlap_duration
                })

print(f"\nFound {len(overlaps)} overlapping kernel pairs:\n")

for idx, ov in enumerate(overlaps[:10]):  # Show first 10
    e1, e2 = ov['e1'], ov['e2']
    print(f"{idx+1}. {e1['type']} [{e1['start']:.2f}-{e1['end']:.2f}ms]")
    print(f"   overlaps with")
    print(f"   {e2['type']} [{e2['start']:.2f}-{e2['end']:.2f}ms]")
    print(f"   Overlap duration: {ov['overlap_ms']:.2f}ms")
    print()

if len(overlaps) > 0:
    total_overlap = sum(ov['overlap_ms'] for ov in overlaps)
    print(f"✓ SUCCESS: Found {len(overlaps)} overlaps!")
    print(f"  Total overlap time: {total_overlap:.2f}ms")
    print(f"\n  This confirms double buffering enables true compute-comm overlap!")
else:
    print("✗ No overlap detected - kernels are still sequential")

print("="*70)
