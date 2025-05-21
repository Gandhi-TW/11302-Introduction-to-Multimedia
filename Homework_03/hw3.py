import cv2
import numpy as np
import math
import os
import time
import matplotlib.pyplot as plt

def compute_psnr(original, predicted):
    mse = np.mean((original.astype(np.float32) - predicted.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def full_search(ref_img, target_img, mb_size, p):
    h, w, c = ref_img.shape
    mv_field = []
    predicted = np.zeros_like(target_img)
    total_sad = 0

    for y in range(0, h, mb_size):
        for x in range(0, w, mb_size):
            min_sad = float('inf')
            best_dx, best_dy = 0, 0
            current_block = target_img[y:y+mb_size, x:x+mb_size, :].astype(np.int16)

            dx_min = max(-p, -x)
            dx_max = min(p, (w - mb_size) - x)
            dy_min = max(-p, -y)
            dy_max = min(p, (h - mb_size) - y)

            for dy in range(dy_min, dy_max + 1):
                for dx in range(dx_min, dx_max + 1):
                    ref_block = ref_img[y+dy:y+dy+mb_size, x+dx:x+dx+mb_size, :].astype(np.int16)
                    sad = np.sum(np.abs(current_block - ref_block))
                    if sad < min_sad:
                        min_sad = sad
                        best_dx, best_dy = dx, dy

            mv_field.append((best_dx, best_dy))
            predicted[y:y+mb_size, x:x+mb_size, :] = ref_img[y+best_dy:y+best_dy+mb_size, x+best_dx:x+best_dx+mb_size, :]
            total_sad += min_sad

    residual = cv2.absdiff(target_img, predicted)
    psnr = compute_psnr(target_img, predicted)
    return mv_field, predicted, residual, total_sad, psnr

def logarithmic_search(ref_img, target_img, mb_size, p):
    h, w, c = ref_img.shape
    mv_field = []
    predicted = np.zeros_like(target_img)
    total_sad = 0

    for y in range(0, h, mb_size):
        for x in range(0, w, mb_size):
            current_block = target_img[y:y+mb_size, x:x+mb_size, :].astype(np.int16)
            dx_center, dy_center = 0, 0
            step = p // 2

            while step >= 1:
                min_sad = float('inf')
                best_dx, best_dy = dx_center, dy_center

                for dy_offset in [-step, 0, step]:
                    for dx_offset in [-step, 0, step]:
                        dx_candidate = dx_center + dx_offset
                        dy_candidate = dy_center + dy_offset

                        if abs(dx_candidate) > p or abs(dy_candidate) > p:
                            continue

                        ref_x = x + dx_candidate
                        ref_y = y + dy_candidate
                        if ref_x < 0 or ref_x + mb_size > w or ref_y < 0 or ref_y + mb_size > h:
                            continue

                        ref_block = ref_img[ref_y:ref_y+mb_size, ref_x:ref_x+mb_size, :].astype(np.int16)
                        sad = np.sum(np.abs(current_block - ref_block))
                        if sad < min_sad:
                            min_sad = sad
                            best_dx, best_dy = dx_candidate, dy_candidate

                dx_center, dy_center = best_dx, best_dy
                step //= 2

            mv_field.append((dx_center, dy_center))
            predicted[y:y+mb_size, x:x+mb_size, :] = ref_img[y+dy_center:y+dy_center+mb_size, x+dx_center:x+dx_center+mb_size, :]
            total_sad += np.sum(np.abs(current_block - ref_img[y+dy_center:y+dy_center+mb_size, x+dx_center:x+dx_center+mb_size, :].astype(np.int16)))

    residual = cv2.absdiff(target_img, predicted)
    psnr = compute_psnr(target_img, predicted)
    return mv_field, predicted, residual, total_sad, psnr

def three_step_search(ref_img, target_img, mb_size, p):
    h, w, c = ref_img.shape
    mv_field = []
    predicted = np.zeros_like(target_img)
    total_sad = 0
    steps = [p//2, p//4, p//8]

    for y in range(0, h, mb_size):
        for x in range(0, w, mb_size):
            current_block = target_img[y:y+mb_size, x:x+mb_size, :].astype(np.int16)
            dx_center, dy_center = 0, 0

            for step in steps:
                min_sad = float('inf')
                best_dx, best_dy = dx_center, dy_center

                for dy_offset in [-step, 0, step]:
                    for dx_offset in [-step, 0, step]:
                        dx_candidate = dx_center + dx_offset
                        dy_candidate = dy_center + dy_offset

                        if abs(dx_candidate) > p or abs(dy_candidate) > p:
                            continue

                        ref_x = x + dx_candidate
                        ref_y = y + dy_candidate
                        if ref_x < 0 or ref_x + mb_size > w or ref_y < 0 or ref_y + mb_size > h:
                            continue

                        ref_block = ref_img[ref_y:ref_y+mb_size, ref_x:ref_x+mb_size, :].astype(np.int16)
                        sad = np.sum(np.abs(current_block - ref_block))
                        if sad < min_sad:
                            min_sad = sad
                            best_dx, best_dy = dx_candidate, dy_candidate

                dx_center, dy_center = best_dx, best_dy

            mv_field.append((dx_center, dy_center))
            predicted[y:y+mb_size, x:x+mb_size, :] = ref_img[y+dy_center:y+dy_center+mb_size, x+dx_center:x+dx_center+mb_size, :]
            total_sad += np.sum(np.abs(current_block - ref_img[y+dy_center:y+dy_center+mb_size, x+dx_center:x+dx_center+mb_size, :].astype(np.int16)))

    residual = cv2.absdiff(target_img, predicted)
    psnr = compute_psnr(target_img, predicted)
    return mv_field, predicted, residual, total_sad, psnr

def draw_motion_vectors(image, mv_field, mb_size):
    drawn = image.copy()
    h, w, _ = image.shape
    idx = 0
    for y in range(0, h, mb_size):
        for x in range(0, w, mb_size):
            if idx >= len(mv_field):
                break
            dx, dy = mv_field[idx]
            cx = x + mb_size // 2
            cy = y + mb_size // 2
            end_x = cx + dx
            end_y = cy + dy
            cv2.arrowedLine(drawn, (cx, cy), (end_x, end_y), (0, 0, 255), 1, tipLength=0.2)
            idx += 1
    return drawn

def main():
    os.makedirs('out', exist_ok=True)
    
    # Problem 1
    ref = cv2.imread('img/008.jpg')
    target = cv2.imread('img/009.jpg')
    assert ref.shape == target.shape, "Images must have the same shape"
    
    for method in ['full', '2d_log']:
        for p in [8, 16]:
            for bs in [8, 16]:
                print(f"Processing: method={method}, p={p}, block_size={bs}")
                if method == 'full':
                    mv, pred, residual, sad, psnr = full_search(ref, target, bs, p)
                elif method == '2d_log':
                    mv, pred, residual, sad, psnr = logarithmic_search(ref, target, bs, p)
                
                cv2.imwrite(f'out/{method}_predicted_r{p}_b{bs}.jpg', pred)
                mv_img = draw_motion_vectors(target, mv, bs)
                cv2.imwrite(f'out/{method}_motion_vector_r{p}_b{bs}.jpg', mv_img)
                cv2.imwrite(f'out/{method}_residual_r{p}_b{bs}.jpg', residual)
                with open('out/results.txt', 'a') as f:
                    f.write(f"{method}, p={p}, bs={bs}: SAD={sad}, PSNR={psnr}\n")

    # Problem 2
    ref_img = cv2.imread('img/000.jpg')
    sads = {'full': [], '2d_log': [], 'three_step': []}
    psnrs = {'full': [], '2d_log': [], 'three_step': []}
    
    for i in range(1, 18):
        target_path = f'img/{i:03d}.jpg'
        target = cv2.imread(target_path)
        if target is None:
            continue
        
        for method in ['full', '2d_log', 'three_step']:
            start = time.time()
            if method == 'full':
                mv, pred, residual, sad, psnr = full_search(ref_img, target, 16, 8)
            elif method == '2d_log':
                mv, pred, residual, sad, psnr = logarithmic_search(ref_img, target, 16, 8)
            else:
                mv, pred, residual, sad, psnr = three_step_search(ref_img, target, 16, 8)
            
            sads[method].append(sad)
            psnrs[method].append(psnr)
            print(f"Method {method} took {time.time() - start:.2f}s for frame {i}")

    # Plotting code would be added here (not implemented in this code)
    frames = list(range(1, 18))  # Frame numbers 001.jpg to 017.jpg

    # SAD Curve
    plt.figure(figsize=(10, 6))
    plt.plot(frames, sads['full'], 'r-o', label='Full Search')
    plt.plot(frames, sads['2d_log'], 'g--s', label='2D-Log')
    plt.plot(frames, sads['three_step'], 'b-.^', label='Three-Step')
    plt.xlabel("Frame Number")
    plt.ylabel("Total SAD")
    plt.title("Total SAD Across Frames (p=8, Macroblock 16x16)")
    plt.xticks(frames)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('out/sad_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # PSNR Curve
    plt.figure(figsize=(10, 6))
    plt.plot(frames, psnrs['full'], 'r-o', label='Full Search')
    plt.plot(frames, psnrs['2d_log'], 'g--s', label='2D-Log')
    plt.plot(frames, psnrs['three_step'], 'b-.^', label='Three-Step')
    plt.xlabel("Frame Number")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Across Frames (p=8, Macroblock 16x16)")
    plt.xticks(frames)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('out/psnr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Problem 3
    ref = cv2.imread('img/008.jpg')
    target = cv2.imread('img/012.jpg')
    mv, pred, residual, sad, psnr = logarithmic_search(ref, target, 16, 8)
    with open('out/results_q3.txt', 'w') as f:
        f.write(f"SAD: {sad}, PSNR: {psnr}\n")
    
    # Problem 4
    for p in [8, 16]:
        for method in ['full', '2d_log', 'three_step']:
            start = time.time()
            if method == 'full':
                full_search(ref, target, 16, p)
            elif method == '2d_log':
                logarithmic_search(ref, target, 16, p)
            else:
                three_step_search(ref, target, 16, p)
            elapsed = time.time() - start
            print(f"Method {method} with p={p} took {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
