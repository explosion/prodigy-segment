from typing import List

from prodigy.components.preprocess import fetch_media
from prodigy.components.stream import get_stream
from prodigy.core import Arg, recipe
from prodigy.protocols import ControllerComponentsDict
from prodigy.types import LabelsType, SourceType, TaskType
from prodigy.util import log


HTML = """
<link
  rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/all.min.css"
  integrity="sha512-1sCRPdkRXhBV2PBLUdRb4tMg1w2YPf37qatUFeS7zlBy7jJI8Lf4VHwWfZZfpXtYSLy85pkm9GaYVYMfw5BC1A=="
  crossorigin="anonymous"
  referrerpolicy="no-referrer"
/>
<button id="refreshButton" onclick="refreshData()">
  Refresh Data Stream
  <i
    id="loadingIcon"
    class="fa-solid fa-spinner fa-spin"
    style="display: none;"
  ></i>
</button>
"""

JS = """
function refreshData() {
  document.querySelector('#loadingIcon').style.display = 'inline-block'
  event_data = {
    example: window.prodigy.content,
  }
  window.prodigy
    .event('segment-anything', event_data)
    .then(updated_example => {
      console.log('Updating Current Example with new data:', updated_example)
      window.prodigy.update(updated_example)
      document.querySelector('#loadingIcon').style.display = 'none'
    })
    .catch(err => {
      console.error('Error in Event Handler:', err)
    })
}
"""

def event_hook(ctrl, *, example: dict):
    print(example)
    example['text'] = 'DINNOSAURRRHEAADDD'
    return example

@recipe(
    "segment.image.manual",
    # fmt: off
    dataset=Arg(help="Dataset to save annotations to"),
    source=Arg(help="Data to annotate (directory of images, file path or '-' to read from standard input)"),
    label=Arg("--label", "-l", help="Comma-separated label(s) to annotate or text file with one label per line"),
    loader=Arg("--loader", "-lo", help="Loader if source is not directory of images"),
    exclude=Arg("--exclude", "-e", help="Comma-separated list of dataset IDs whose annotations to exclude"),
    darken=Arg("--darken", "-D", help="Darken image to make boxes stand out more"),
    width=Arg("--width", "-w", help="Default width of the annotation card and space for the image (in px)"),
    no_fetch=Arg("--no-fetch", "-NF", help="Don't fetch images as base64"),
    remove_base64=Arg("--remove-base64", "-R", help="Remove base64-encoded image data before storing example in the DB. (Caution: if enabled, make sure to keep original files!)")
    # fmt: on
)
def segment_image_manual(
    dataset: str,
    source: SourceType,
    label: LabelsType,
    loader: str = "images",
    exclude: List[str] = [],
    darken: bool = False,
    width: int = 675,
    no_fetch: bool = False,
    remove_base64: bool = False,
) -> ControllerComponentsDict:
    """
    Manually annotate images by drawing rectangular bounding boxes or polygon
    shapes on the image.
    """
    log("RECIPE: Starting recipe image.manual", locals())
    stream = get_stream(
        source,
        loader=loader,
        dedup=True,
        rehash=True,
        input_key="image",
        is_binary=False,
    )
    if not no_fetch and loader != "image-server":
        stream.apply(fetch_media, stream=stream, input_keys=["image"])

    def before_db(examples: List[TaskType]) -> List[TaskType]:
        # Remove all data URIs before storing example in the database
        for eg in examples:
            if eg["image"].startswith("data:"):
                eg["image"] = eg.get("path")
        return examples

    blocks = [{"view_id": "image_manual"}, {"view_id": "html", "html_template": HTML}]

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "before_db": before_db if remove_base64 else None,
        "exclude": exclude,
        "config": {
            "labels": label,
            "blocks": blocks,
            "darken_image": 0.3 if darken else 0,
            "custom_theme": {"cardMaxWidth": width},
            "exclude_by": "input",
            "auto_count_stream": True,
            "javascript": JS
        },
        "event_hooks": {
            "segment-anything": event_hook
        }
    }
