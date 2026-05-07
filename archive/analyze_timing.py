def parse_time_ms(t):
    """Convert time string to ms"""
    if 's' in t and 'ms' not in t:
        return float(t.replace('s', '')) * 1000
    elif 'ms' in t:
        return float(t.replace('ms', ''))
    return 0

# Overlap data (GEMM, NCCL alternating)
overlap_data = """
1.07471s 6.4939ms NCCL
1.08122s 3.8764ms GEMM
1.08510s 6.6443ms NCCL
1.09177s 3.8668ms GEMM
1.09564s 6.6908ms NCCL
1.10235s 3.8723ms GEMM
1.10622s 6.5215ms NCCL
1.11312s 3.8734ms GEMM
1.11699s 6.6448ms NCCL
1.12366s 3.8736ms GEMM
"""

print("Overlap Mode Timeline Analysis:")
print("="*60)

lines = [l.strip() for l in overlap_data.strip().split('\n') if l.strip()]
for i, line in enumerate(lines):
    parts = line.split()
    start = parse_time_ms(parts[0])
    duration = parse_time_ms(parts[1])
    ktype = parts[2]
    end = start + duration
    
    print(f"{i}: {ktype:4} starts at {start:8.2f}ms, ends at {end:8.2f}ms (duration {duration:.2f}ms)")
    
    if i > 0:
        prev_parts = lines[i-1].split()
        prev_start = parse_time_ms(prev_parts[0])
        prev_duration = parse_time_ms(prev_parts[1])
        prev_end = prev_start + prev_duration
        
        gap = start - prev_end
        if gap < 0:
            print(f"   ✓ OVERLAP! Current starts {-gap:.2f}ms before previous ends")
        else:
            print(f"   ✗ GAP of {gap:.2f}ms (sequential execution)")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("If there's overlap, GEMM should start BEFORE previous NCCL ends")
print("="*60)
