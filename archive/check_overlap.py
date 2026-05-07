import re

with open('overlap_nvprof.txt', 'r') as f:
    lines = f.readlines()

kernels = []
for line in lines:
    if 'volta_sgemm' in line or 'ncclDevKernel' in line:
        parts = line.split()
        if len(parts) >= 2:
            start = parts[0]
            duration = parts[1]
            stream = None
            for i, p in enumerate(parts):
                if p == 'Stream' and i+1 < len(parts):
                    stream = parts[i+1]
                    break
            
            ktype = 'GEMM' if 'sgemm' in line else 'NCCL'
            kernels.append({
                'start': start,
                'duration': duration,
                'stream': stream,
                'type': ktype
            })

print("First 20 kernels in overlap mode:")
print(f"{'Start':<12} {'Duration':<10} {'Stream':<8} {'Type':<6}")
print("-" * 40)
for k in kernels[:20]:
    print(f"{k['start']:<12} {k['duration']:<10} {k['stream']:<8} {k['type']:<6}")

# Check for overlap
print("\n\nChecking for overlap...")
for i in range(min(10, len(kernels)-1)):
    k1 = kernels[i]
    k2 = kernels[i+1]
    print(f"\n{k1['type']} (stream {k1['stream']}) at {k1['start']}")
    print(f"{k2['type']} (stream {k2['stream']}) at {k2['start']}")
    if k1['stream'] != k2['stream']:
        print("  -> Different streams, checking overlap...")
    else:
        print("  -> Same stream, sequential")
