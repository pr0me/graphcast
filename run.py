import dataclasses
import datetime
import math
from google.cloud import storage
from typing import Optional
import haiku as hk
import jax
import jax.experimental
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning

print("[*] IMPORTS SUCCESSFUL")

# PLOTTING FUNCTIONS
def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax),
        ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols,
                                plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"))
        images.append(im)
    plt.show()

    # def update(frame):
    #     if "time" in first_data.dims:
    #         td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
    #         figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    #     else:
    #         figure.suptitle(fig_title, fontsize=16)
    #     for im, (plot_data, norm, cmap) in zip(images, data.values()):
    #         im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

    #     ani = animation.FuncAnimation(
    #         fig=figure, func=update, frames=max_steps, interval=250)
    #     plt.close(figure.number)
    #     return HTML(ani.to_jshtml())

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"

# params_file_options = [
#     name for blob in gcs_bucket.list_blobs(prefix=(dir_prefix+"params/"))
#     if (name := blob.name.removeprefix(dir_prefix+"params/"))]
# print(params_file_options)
# ['GenCast 0p25deg <2019.npz', 'GenCast 0p25deg Operational <2022.npz', 'GenCast 1p0deg <2019.npz', 'GenCast 1p0deg Mini <2019.npz']

params_file = 'GenCast 1p0deg <2019.npz'

def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))

# dataset_file_options = [
#     name for blob in gcs_bucket.list_blobs(prefix=(dir_prefix + "dataset/"))
#     if (name := blob.name.removeprefix(dir_prefix+"dataset/"))]
# print(dataset_file_options)
dataset_file = 'source-era5_date-2019-03-29_res-1.0_levels-13_steps-12.nc'

with gcs_bucket.blob(dir_prefix+f"dataset/{dataset_file}").open("rb") as f:
    example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))

print()

plot_example_variable = "2m_temperature"
plot_example_level = 500
plot_example_robust = True
plot_example_max_steps = example_batch.dims["time"]

plot_size = 7

data = {
    " ": scale(select(example_batch, plot_example_variable, plot_example_level, plot_example_max_steps),
                robust=plot_example_robust),
}
fig_title = plot_example_variable
if "level" in example_batch[plot_example_variable].coords:
    fig_title += f" at {plot_example_level} hPa"

print("[*] LOADED DATASET")

# plot_data(data, fig_title, plot_size, plot_example_robust)

# assert source == "Checkpoint"
with gcs_bucket.blob(dir_prefix + f"params/{params_file}").open("rb") as f:
    ckpt = checkpoint.load(f, gencast.CheckPoint)
    params = ckpt.params
    state = {}

    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config

    # configure for non-TPU
    denoiser_architecture_config = ckpt.denoiser_architecture_config
    denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
    denoiser_architecture_config.sparse_transformer_config.mask_type = "full"

    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")

print("[*] LOADED CHECKPOINT")
print(f"    Number of local devices {len(jax.local_devices())}")

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", f"{(example_batch.dims['time']-2)*12}h"), # All but 2 input frames.
    **dataclasses.asdict(task_config))

# normalization data
with gcs_bucket.blob(dir_prefix+"stats/diffs_stddev_by_level.nc").open("rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob(dir_prefix+"stats/mean_by_level.nc").open("rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob(dir_prefix+"stats/stddev_by_level.nc").open("rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob(dir_prefix+"stats/min_by_level.nc").open("rb") as f:
    min_by_level = xarray.load_dataset(f).compute()

def construct_wrapped_gencast():
    """Constructs and wraps the GenCast Predictor."""
    predictor = gencast.GenCast(
        sampler_config=sampler_config,
        task_config=task_config,
        denoiser_architecture_config=denoiser_architecture_config,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
    )

    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    predictor = nan_cleaning.NaNCleaner(
        predictor=predictor,
        reintroduce_nans=True,
        fill_value=min_by_level,
        var_to_clean='sea_surface_temperature',
    )

    return predictor


@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
    predictor = construct_wrapped_gencast()
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(inputs, targets, forcings):
    predictor = construct_wrapped_gencast()
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics),
    )


def grads_fn(params, state, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), i, t, f
        )
        return loss, (diagnostics, next_state)

    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True
    )(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

loss_fn_jitted = jax.jit(
    lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
)
grads_fn_jitted = jax.jit(grads_fn)
run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)

# pmapped version for running in parallel.
run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

# INFERENCE

# autoregressive rollout
print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

num_ensemble_members = 8 # @param int
rng = jax.random.PRNGKey(0)
# We fold-in the ensemble member, this way the first N members should always
# match across different runs which use take the same inputs, regardless of
# total ensemble size.
rngs = np.stack(
    [jax.random.fold_in(rng, i) for i in range(num_ensemble_members)], axis=0)

chunks = []
for chunk in rollout.chunked_prediction_generator_multiple_runs(
    # Use pmapped version to parallelise across devices.
    predictor_fn=run_forward_pmap,
    rngs=rngs,
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
    num_steps_per_chunk = 1,
    num_samples = num_ensemble_members,
    pmap_devices=jax.local_devices()
    ):
    chunks.append(chunk)
predictions = xarray.combine_by_coords(chunks)

plot_pred_variable = "2m_temperature"
plot_pred_level = 500
plot_pred_robust = True
plot_pred_max_steps = predictions.dims["time"]
plot_pred_samples = num_ensemble_members

# Plot prediction samples and diffs

plot_size = 5
plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps.value)

fig_title = plot_pred_variable.value
if "level" in predictions[plot_pred_variable.value].coords:
    fig_title += f" at {plot_pred_level.value} hPa"

for sample_idx in range(plot_pred_samples.value):
    data = {
        "Targets": scale(select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
        "Predictions": scale(select(predictions.isel(sample=sample_idx), plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
        "Diff": scale((select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps) -
                select(predictions.isel(sample=sample_idx), plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
                robust=plot_pred_robust.value, center=0),
    }
    plot_data(data, fig_title + f", Sample {sample_idx}", plot_size, plot_pred_robust.value)

# Plot ensemble mean and CRPS

def crps(targets, predictions, bias_corrected = True):
    if predictions.sizes.get("sample", 1) < 2:
        raise ValueError(
        "predictions must have dim 'sample' with size at least 2.")
    sum_dims = ["sample", "sample2"]
    preds2 = predictions.rename({"sample": "sample2"})
    num_samps = predictions.sizes["sample"]
    num_samps2 = (num_samps - 1) if bias_corrected else num_samps
    mean_abs_diff = np.abs(
        predictions - preds2).sum(
          dim=sum_dims, skipna=False) / (num_samps * num_samps2)
    mean_abs_err = np.abs(targets - predictions).sum(dim="sample", skipna=False) / num_samps
    return mean_abs_err - 0.5 * mean_abs_diff


plot_size = 5
plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps.value)

fig_title = plot_pred_variable.value
if "level" in predictions[plot_pred_variable.value].coords:
    fig_title += f" at {plot_pred_level.value} hPa"

data = {
    "Targets": scale(select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
    "Ensemble Mean": scale(select(predictions.mean(dim=["sample"]), plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
    "Ensemble CRPS": scale(crps((select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
                        select(predictions, plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
                        robust=plot_pred_robust.value, center=0),
}
plot_data(data, fig_title, plot_size, plot_pred_robust.value)