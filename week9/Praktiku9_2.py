import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def latihan_2():
    # Buat bentuk sederhana untuk analisis boundary
    shapes = []
    
    # 1. Persegi (untuk chain code sederhana)
    square = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(square, (20, 20), (80, 80), 255, 1)  # Outline only
    shapes.append(('Square', square))
    
    # 2. Lingkaran (untuk Fourier analysis)
    circle = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(circle, (50, 50), 30, 255, 1)
    shapes.append(('Circle', circle))
    
    # 3. Segitiga (bentuk sederhana tapi non-simetris)
    triangle = np.zeros((100, 100), dtype=np.uint8)
    pts = np.array([[50, 20], [20, 80], [80, 80]])
    cv2.polylines(triangle, [pts], True, 255, 1)
    shapes.append(('Triangle', triangle))
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row, (name, shape) in enumerate(shapes):
        # Dapatkan contour
        contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contours[0]
        
        # Simplifikasi contour untuk chain code
        contour_simple = []
        for i in range(len(contour)):
            if i == 0 or not np.array_equal(contour[i][0], contour[i-1][0]):
                contour_simple.append(contour[i][0])
        contour_simple = np.array(contour_simple)
        
        # 1. Original shape
        axes[row, 0].imshow(shape, cmap='gray')
        axes[row, 0].set_title(f'{name}\nOriginal Shape')
        axes[row, 0].axis('off')
        
        # 2. Freeman Chain Code (8-directional)
        def freeman_chain_code_8dir(points):
            """Calculate 8-directional Freeman chain code"""
            if len(points) < 2:
                return []
            
            directions_8 = [
                (1, 0),   # 0: East
                (1, 1),   # 1: Southeast
                (0, 1),   # 2: South
                (-1, 1),  # 3: Southwest
                (-1, 0),  # 4: West
                (-1, -1), # 5: Northwest
                (0, -1),  # 6: North
                (1, -1)   # 7: Northeast
            ]
            
            chain_code = []
            for i in range(len(points)):
                current = points[i]
                next_point = points[(i + 1) % len(points)]
                
                dx = next_point[0] - current[0]
                dy = next_point[1] - current[1]
                
                # Cari direction yang cocok
                for dir_code, (dir_dx, dir_dy) in enumerate(directions_8):
                    if dx == dir_dx and dy == dir_dy:
                        chain_code.append(dir_code)
                        break
            
            return chain_code
        
        chain_code = freeman_chain_code_8dir(contour_simple)
        
        # Visualisasi chain code
        chain_display = np.zeros((100, 100), dtype=np.uint8)
        for i, point in enumerate(contour_simple):
            cv2.circle(chain_display, tuple(point), 1, 255, -1)
            if i < len(chain_code):
                # Gambar arrow untuk direction
                dir_code = chain_code[i]
                directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
                dx, dy = directions[dir_code]
                end_point = (point[0] + dx*3, point[1] + dy*3)
                cv2.arrowedLine(chain_display, tuple(point), end_point, 200, 1)
        
        axes[row, 1].imshow(chain_display, cmap='gray')
        axes[row, 1].set_title(f'Chain Code\n{len(chain_code)} directions')
        axes[row, 1].axis('off')
        
        # 3. Fourier Descriptors
        def compute_fourier_descriptors(points, num_descriptors=20):
            """Compute Fourier descriptors from boundary points"""
            # Convert points to complex numbers
            z = points[:, 0] + 1j * points[:, 1]
            
            # Apply Fourier Transform
            fd = np.fft.fft(z)
            
            # Get magnitude spectrum
            fd_magnitude = np.abs(fd)
            
            # Normalize (divide by DC component)
            if fd_magnitude[0] != 0:
                fd_normalized = fd_magnitude / fd_magnitude[0]
            else:
                fd_normalized = fd_magnitude
            
            return fd, fd_normalized
        
        # Pastikan kita punya cukup points untuk Fourier analysis
        if len(contour_simple) > 10:
            # Resample points untuk uniform sampling
            num_samples = min(256, len(contour_simple))
            indices = np.linspace(0, len(contour_simple)-1, num_samples).astype(int)
            sampled_points = contour_simple[indices]
            
            fd, fd_norm = compute_fourier_descriptors(sampled_points, 20)
            
            # Plot Fourier descriptors magnitude
            n = len(fd_norm)
            frequencies = np.fft.fftfreq(n)
            
            axes[row, 2].stem(frequencies[:n//2], fd_norm[:n//2])
            axes[row, 2].set_title('Fourier Descriptors\nMagnitude Spectrum')
            axes[row, 2].set_xlabel('Frequency')
            axes[row, 2].set_ylabel('Normalized Magnitude')
            axes[row, 2].grid(True, alpha=0.3)
            
            # 4. Shape reconstruction from Fourier descriptors
            def reconstruct_shape(fd, num_coeffs):
                """Reconstruct shape using limited Fourier coefficients"""
                fd_recon = fd.copy()
                # Keep only low frequency components
                fd_recon[num_coeffs:-num_coeffs] = 0
                z_recon = np.fft.ifft(fd_recon)
                return z_recon.real, z_recon.imag
            
            # Rekonstruksi dengan jumlah coefficients berbeda
            for coeff_idx, num_coeffs in enumerate([5, 10]):
                x_recon, y_recon = reconstruct_shape(fd, num_coeffs)
                
                recon_img = np.zeros((100, 100), dtype=np.uint8)
                points_recon = np.column_stack([x_recon.astype(int), y_recon.astype(int)])
                
                # Gambar reconstructed shape
                for i in range(len(points_recon)):
                    start_point = points_recon[i]
                    end_point = points_recon[(i + 1) % len(points_recon)]
                    cv2.line(recon_img, tuple(start_point), tuple(end_point), 255, 1)
                
                axes[row, 3].imshow(recon_img, cmap='gray')
                axes[row, 3].set_title(f'Reconstructed\n{num_coeffs} coefficients')
                axes[row, 3].axis('off')
        
        else:
            axes[row, 2].text(0.5, 0.5, 'Not enough points\nfor Fourier analysis', 
                            ha='center', va='center', transform=axes[row, 2].transAxes)
            axes[row, 2].axis('off')
            axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analisis chain code
    print("CHAIN CODE ANALYSIS")
    print("=" * 40)
    
    for name, shape in shapes:
        contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contours[0]
        
        contour_simple = []
        for i in range(len(contour)):
            if i == 0 or not np.array_equal(contour[i][0], contour[i-1][0]):
                contour_simple.append(contour[i][0])
        
        chain_code = freeman_chain_code_8dir(contour_simple)
        
        print(f"\n{name}:")
        print(f"  Number of chain codes: {len(chain_code)}")
        print(f"  First 10 codes: {chain_code[:10]}")
        print(f"  Direction frequency:")
        for dir_code in range(8):
            count = sum(1 for code in chain_code if code == dir_code)
            if count > 0:
                directions = ['East', 'SE', 'South', 'SW', 'West', 'NW', 'North', 'NE']
                print(f"    {directions[dir_code]}: {count}")
    
    # Fourier descriptors untuk shape matching
    print("\nFOURIER DESCRIPTORS FOR SHAPE MATCHING")
    print("=" * 45)
    
    # Hitung similarity antara shapes menggunakan Fourier descriptors
    if len(shapes) >= 3:
        fd_list = []
        shape_names_list = []
        
        for name, shape in shapes[:3]:  # Ambil 3 shapes pertama
            contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = contours[0]
            
            contour_simple = []
            for i in range(len(contour)):
                if i == 0 or not np.array_equal(contour[i][0], contour[i-1][0]):
                    contour_simple.append(contour[i][0])
            
            # Resample
            num_samples = 64
            indices = np.linspace(0, len(contour_simple)-1, num_samples).astype(int)
            sampled_points = np.array(contour_simple)[indices]
            
            fd, fd_norm = compute_fourier_descriptors(sampled_points, 10)
            fd_list.append(fd_norm[:10])  # Ambil 10 coefficients pertama
            shape_names_list.append(name)
        
        # Hitung Euclidean distance antara Fourier descriptors
        print("Shape similarity (Euclidean distance between Fourier descriptors):")
        for i in range(len(fd_list)):
            for j in range(i + 1, len(fd_list)):
                distance = np.sqrt(np.sum((fd_list[i] - fd_list[j]) ** 2))
                print(f"  {shape_names_list[i]} vs {shape_names_list[j]}: {distance:.4f}")
        
        print("\nInterpretation:")
        print("• Smaller distance = more similar shapes")
        print("• Larger distance = less similar shapes")
        print("• Fourier descriptors are invariant to rotation and scale")

# Jalankan latihan 2
latihan_2()