import matplotlib.pyplot as plt
import numpy as np

# 1. Generate Dummy Training Data (Replace this with your actual log arrays)
epochs = np.arange(1, 101)
# Simulating a smooth loss curve dropping over time
loss = 7.5 * np.exp(-0.05 * epochs) + 1.5 + np.random.normal(0, 0.1, len(epochs))
# Perplexity is exp(Loss), but we'll scale it slightly for visual framing
perplexity = np.exp(loss / 2.5) 

# 2. Configure High-End Dark Mode Aesthetics
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300) # High-res output
fig.patch.set_facecolor('#0d1117') # Exact GitHub Dark Mode background
ax1.set_facecolor('#0d1117')

# 3. Plot Cross-Entropy Loss (Left Axis)
color1 = '#00f2fe' # Neon Cyan
ax1.set_xlabel('Training Iterations', fontsize=12, fontweight='bold', labelpad=15)
ax1.set_ylabel(r'Cross-Entropy Loss ($\mathcal{L}$)', color=color1, fontsize=14, fontweight='bold')
line1, = ax1.plot(epochs, loss, color=color1, linewidth=2.5, alpha=0.9, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(color='#30363d', linestyle='--', linewidth=0.5, alpha=0.7)

# 4. Plot Perplexity (Right Axis)
ax2 = ax1.twinx()  
color2 = '#fe007a' # Neon Pink
ax2.set_ylabel(r'Perplexity ($e^\mathcal{L}$)', color=color2, fontsize=14, fontweight='bold', rotation=270, labelpad=25)
line2, = ax2.plot(epochs, perplexity, color=color2, linewidth=2.5, alpha=0.9, label='Perplexity')
ax2.tick_params(axis='y', labelcolor=color2)

# 5. Polish and Export
plt.title('Llama 3 Core: Training Convergence', fontsize=16, fontweight='bold', color='white', pad=20)
fig.tight_layout()

# Combine legends from both axes
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', facecolor='#161b22', edgecolor='#30363d', fontsize=10)

# Save the high-resolution image
plt.savefig('assets/training_convergence.png', bbox_inches='tight', facecolor=fig.get_facecolor())
print("Graph successfully generated and saved to 'assets/training_convergence.png'")