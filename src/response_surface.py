from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

fig, axes = plt.subplots(ncols=3, figsize=(12, 4), layout='constrained')
titles = ['Cluster\\,1', 'Cluster\\,2', 'Entire\\,Sample']
for i, data in enumerate([data_c1, data_c2, data_all]):
    X = data[['Percent Extraction', 'TDS__1']].to_numpy()
    y = data['Liking'].to_numpy()

    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)

    x_grid = np.linspace(14, 28, 100)
    y_grid = np.linspace(0.9, 1.7, 100)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    # Reshape for prediction
    Z = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

    Z_pred = poly_model.predict(Z)
    Z_pred = Z_pred.reshape(X_mesh.shape) # Reshape back to grid for plotting

    p = axes[i].contourf(X_mesh, Y_mesh, Z_pred, levels=1000, cmap=sns.color_palette("viridis", as_cmap=True))
    p = axes[i].contour(X_mesh, Y_mesh, Z_pred, levels=20, linewidths=0.5, alpha=0.5) # cmap=sns.color_palette("viridis", as_cmap=True))
    axes[i].scatter(data['Percent Extraction'], data['TDS__1'], c=data['Liking'], cmap='viridis')

    # Original BCC lines for reference
    axes[i].axvline(x=18, color='black', linestyle='--', lw=0.5)
    axes[i].axvline(x=22, color='black', linestyle='--', lw=0.5)
    axes[i].axhline(y=1.15, color='black', linestyle='--', lw=0.5)
    axes[i].axhline(y=1.35, color='black', linestyle='--', lw=0.5)

    # axes[i].set_xlabel('Percent Extraction')
    # axes[i].set_ylabel('TDS')
    axes[i].set_title(f"$\\bf{{{titles[i]}}}$")

# Add colorbar
norm = plt.Normalize(data['Liking'].min(), data['Liking'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='Liking')
fig.supxlabel('Percent Extraction')
fig.supylabel('TDS')
plt.show()