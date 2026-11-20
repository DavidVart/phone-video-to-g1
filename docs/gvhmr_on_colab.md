# Running GVHMR on today's Google Colab

The [GVHMR Colab notebook](https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj?usp=sharing) was written for Python 3.10 and CUDA 12. Colab now runs Python 3.13 and CUDA 13, so the notebook's own setup fails. The steps below install GVHMR into a separate Python 3.10 environment inside the Colab session instead. They worked on a free T4 GPU in November 2026.

A Colab session is temporary: when it disconnects, its files are deleted and the steps must be run again (about 7 minutes). Download the result as soon as GVHMR finishes.

1. Open the notebook, sign in to Google, and confirm the runtime is a T4 GPU.
2. Run the first cell under **Environment Preparation**, which downloads GVHMR. It prints `ERROR: pytorch3d-0.7.6-cp310... is not a supported wheel`; ignore that. Skip the **Install DPVO** cell: it stops with `CUDA_HOME for cuda 12.x not found`, and a still camera does not need it.
3. Add a code cell and create the Python 3.10 environment (about 10 seconds):

   ```
   !pip -q install uv && uv venv --seed --python 3.10 /content/gv310
   ```

4. Add a code cell and install GVHMR's packages (about 4 minutes; it ends with `INSTALL_OK`):

   ```
   !cd /content/GVHMR && /content/gv310/bin/python -m pip install -q numpy==1.23.5 scipy six cython && /content/gv310/bin/python -m pip install -q --no-build-isolation chumpy cython_bbox && /content/gv310/bin/python -m pip install -q -r requirements.txt && /content/gv310/bin/python -m pip install -q -e . && echo INSTALL_OK
   ```

5. Run the notebook's **Data Preparation** cell as it is (about 1 minute, 5.6 GB of models). To test the install, run the built-in example (about 3.5 minutes); the result is `/content/GVHMR/outputs/demo/tennis/hmr4d_results.pt`:

   ```
   !cd /content/GVHMR && /content/gv310/bin/python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
   ```

6. For your own video, add a code cell with the line below and run it. A **Choose Files** button appears under the cell; pick the 30 fps video. The cell runs GVHMR with a still camera (`-s`) and starts the download of the result. A 25-second, 720 × 1280 video took about 8 minutes, 4.5 of them preprocessing.

   ```python
   from google.colab import files; import os; os.chdir('/content'); up = files.upload(); video = next(iter(up)); stem = video.rsplit('.', 1)[0]; get_ipython().system(f'cd /content/GVHMR && /content/gv310/bin/python tools/demo/demo.py --video="/content/{video}" -s'); files.download(f'/content/GVHMR/outputs/demo/{stem}/hmr4d_results.pt')
   ```

   The result is `/content/GVHMR/outputs/demo/<video name without extension>/hmr4d_results.pt`. To start the download again, run `from google.colab import files; files.download('<that path>')` in a new cell.

Notes:

- The video uploaded this way goes into your own Colab session only.
- GVHMR treats results as 30 frames per second, and GMR assumes the same, so convert the video to 30 fps before uploading.
- The Data Preparation cell fetches the SMPL and SMPL-X body models from a third party's Hugging Face copy. The official source is [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/), which needs a free registration and limits use to non-commercial research and education.
- The notebook's own method for a custom video needs a Google Drive link shared with "Anyone with the link"; step 6 avoids that.
