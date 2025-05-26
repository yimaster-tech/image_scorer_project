import os
import gradio as gr
import shutil
from scoring_utils import (
    get_aesthetic_score,
    compute_blur_score,
    get_image_map,
    face_similarity_tools,
    batch_face_similarity,
    batch_clip_similarity,
    export_csv
)
from Config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

output_dir = Config.OUTPUT_DIR
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义项目内的图像存储目录
PROJECT_IMAGE_DIR = os.path.join(output_dir, "uploaded_images")
if not os.path.exists(PROJECT_IMAGE_DIR):
    os.makedirs(PROJECT_IMAGE_DIR)

def image_to_base64(image_path):
    """
    将图片文件转换为 base64 编码的 data URI 字符串。
    Args:
        image_path (str): 图片文件路径。
    Returns:
        str: base64 编码的 data URI 字符串。
    """
    import base64
    with open(image_path, "rb") as img_f:
        encoded = base64.b64encode(img_f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[-1][1:]
        return f"data:image/{ext};base64,{encoded}"

def start(image_folder: str, reference_image_name: str = None):
    """
    启动图像评分流程，计算图像的美学、清晰度、人脸相似、全局人脸相似、CLIP 相似性等数据。
    Args:
        image_folder (str): 图像所在的文件夹路径。
        reference_image_name (str, optional): 参考图像的文件名，用于计算人脸相似度。默认为 None。
    Returns:
        dict: 包含各项评分结果的字典。
    """
    try:
        print(f"📂 加载图像文件夹: {image_folder}")
        image_map = get_image_map(image_folder)
        if not image_map:
            print("❌ 未找到有效的图像文件。")
            return {}
    except Exception as e:
        print(f"❌ 加载图像文件夹失败: {e}")
        return {}

    reference_img = None
    if reference_image_name:
        reference_img_path = os.path.join(image_folder, reference_image_name)
        if os.path.exists(reference_img_path):
            reference_img = image_map.get(reference_image_name)
        else:
            print(f"⚠️ 参考图像 {reference_image_name} 不存在。")

    def score_one(filename):
        img_path = os.path.join(image_folder, filename)
        try:
            aes_score = get_aesthetic_score(img_path)
        except Exception as e:
            print(f"❌ 计算美学分失败: {filename}, {e}")
            aes_score = -1.0
        try:
            blur_score = compute_blur_score(img_path)
        except Exception as e:
            print(f"❌ 计算清晰度失败: {filename}, {e}")
            blur_score = -1.0
        sim_score = -1.0
        if reference_img is not None:
            try:
                sim_score = face_similarity_tools(image_map[filename], reference_img)
            except Exception as e:
                print(f"❌ 计算 {filename} 与参考图像的人脸相似度失败: {e}")
        return (filename, aes_score, blur_score, sim_score)

    results = []
    try:
        with ThreadPoolExecutor() as executor:
            future_to_name = {executor.submit(score_one, filename): filename for filename in image_map.keys()}
            for future in as_completed(future_to_name):
                res = future.result()
                results.append(res)
        # 保证顺序与原始文件一致
        results.sort(key=lambda x: list(image_map.keys()).index(x[0]))
    except Exception as e:
        print(f"❌ 并行评分失败: {e}")
        return {}

    try:
        face_sim_results, avg_face_sim = batch_face_similarity(image_map)
    except Exception as e:
        print(f"❌ 批量人脸相似度计算失败: {e}")
        face_sim_results, avg_face_sim = [], -1.0
    try:
        clip_sim_results = batch_clip_similarity(image_folder, image_map)
    except Exception as e:
        print(f"❌ CLIP 相似度计算失败: {e}")
        clip_sim_results = []
    try:
        csv_path = os.path.join(output_dir, "image_scores.csv")
        export_csv(data=results, csv_path=csv_path)
    except Exception as e:
        print(f"❌ 导出 CSV 失败: {e}")
        csv_path = None
    return {
        "image_scores": results,
        "face_similarity_results": face_sim_results,
        "average_face_similarity": avg_face_sim,
        "clip_similarity_results": clip_sim_results,
        "csv_path": csv_path
    }


def gradio_interface(image_folder, reference_image):
    """
    Gradio 主界面回调函数，处理图片上传、评分及结果展示。
    Args:
        image_folder (list): 上传的图片文件路径列表。
        reference_image (str): 参考图片文件路径。
    Returns:
        tuple: 多项评分和展示内容。
    """
    if not image_folder:
        return "未选择图像文件夹。", None, None, None, None
    if not reference_image:
        return "<span style='color:red;'>❌ 请上传参考图像后再点击开始评分。</span>", "", "", "", "", "", "", None
    # 清空 PROJECT_IMAGE_DIR
    try:
        if os.path.exists(PROJECT_IMAGE_DIR):
            shutil.rmtree(PROJECT_IMAGE_DIR)
        os.makedirs(PROJECT_IMAGE_DIR, exist_ok=True)
        for src_path in image_folder:
            base_name = os.path.basename(src_path)
            # 跳过 DS_Store
            if base_name.endswith(".DS_Store") or 'DS_Store' in base_name:
                continue
            dst_path = os.path.join(PROJECT_IMAGE_DIR, base_name)
            shutil.copy2(src_path, dst_path)
    except Exception as e:
        return f"<span style='color:red;'>❌ 文件复制失败: {e}</span>", "", "", "", "", "", "", None

    # 使用项目内的图像目录
    image_folder_path = PROJECT_IMAGE_DIR
    reference_image_name = os.path.basename(reference_image)
    try:
        if reference_image:
            dst_fullpath = os.path.join(PROJECT_IMAGE_DIR, reference_image_name)
            if not os.path.exists(dst_fullpath):
                shutil.copy2(reference_image, dst_fullpath)
    except Exception as e:
        return f"<span style='color:red;'>❌ 参考图像复制失败: {e}</span>", "", "", "", "", "", "", None

    try:
        results = start(image_folder_path, reference_image_name)
    except Exception as e:
        return f"<span style='color:red;'>❌ 评分流程异常: {e}</span>", "", "", "", "", "", "", None
    if not results:
        return "未找到有效的图像文件。", None, None, None, None

    reference_img_path = os.path.join(image_folder_path, reference_image_name)
    ref_aes = get_aesthetic_score(reference_img_path)
    ref_blur = compute_blur_score(reference_img_path)
    ref_img_data = image_to_base64(reference_img_path)

    popup_html = """
    <style>
      .popup-img:hover { cursor: zoom-in; }
      .popup-overlay {
          display: none;
          position: fixed;
          top: 0; left: 0; width: 100%; height: 100%;
          background: rgba(0,0,0,0.7);
          justify-content: center; align-items: center;
          z-index: 9999;
      }
      .popup-overlay img {
          max-width: 90vw; max-height: 90vh;
          box-shadow: 0 0 10px #fff;
      }
    </style>
    <script>
      function enlargeImage(src) {
          const overlay = document.getElementById('popupOverlay');
          const img = document.getElementById('popupImg');
          img.src = src;
          overlay.style.display = 'flex';
      }
      function closeOverlay() {
          document.getElementById('popupOverlay').style.display = 'none';
      }
    </script>
    <div id="popupOverlay" class="popup-overlay" onclick="closeOverlay()">
      <img id="popupImg" src="" />
    </div>
    """

    reference_html_str = f"""
    <h3>参考图像评分</h3>
    <div style='display:inline-block; margin:10px; position:relative;'>
        <img src="{ref_img_data}" style='width:200px; display:block;' class="popup-img" onclick="enlargeImage(this.src)" />
        <div style='position:absolute; top:4px; left:6px; background-color:rgba(0,0,0,0.5); color:white; padding:2px 4px; font-size:12px;'>美学 {ref_aes:.2f}</div>
        <div style='position:absolute; top:24px; left:6px; background-color:rgba(0,0,0,0.5); color:white; padding:2px 4px; font-size:12px;'>清晰 {ref_blur:.2f}</div>
    </div>
    """
    reference_html_str = popup_html + reference_html_str

    html_str = "<h3>评分图像展示</h3>"
    for name, aes, blur, sim in results["image_scores"]:
        img_path = os.path.join(image_folder_path, name)
        img_data = image_to_base64(img_path)
        html_str += f"""
        <div style='display:inline-block; margin:10px; position:relative;'>
            <img src="{img_data}" style='width:200px; display:block;' class="popup-img" onclick="enlargeImage(this.src)" />
            <div style='position:absolute; top:4px; left:6px; background-color:rgba(0,0,0,0.5); color:white; padding:2px 4px; font-size:12px;'>美学 {aes:.2f}</div>
            <div style='position:absolute; top:24px; left:6px; background-color:rgba(0,0,0,0.5); color:white; padding:2px 4px; font-size:12px;'>清晰 {blur:.2f}</div>
            <div style='position:absolute; top:44px; left:6px; background-color:rgba(0,0,0,0.5); color:white; padding:2px 4px; font-size:12px;'>人脸 {sim:.2f}</div>
        </div>
        """
    image_scores_text = "文件名\t\t美学分\t清晰度\t人脸相似度\n"
    image_scores_text += "\n".join([
        f"{name:<20}\t{aes:.2f}\t{blur:.2f}\t{sim:.2f}"
        for name, aes, blur, sim in results["image_scores"]
    ])

    face_sim_text = "人脸相似度对比（文件1 <--> 文件2）\n"
    face_sim_text += "\n".join([
        f"{name1:<20} <--> {name2:<20} : {sim:.2f}"
        for name1, name2, sim in results["face_similarity_results"]
    ])

    avg_face_sim_text = f"平均人脸相似度: {results['average_face_similarity']:.2f}"

    clip_sim_text = "CLIP 相似度对比（文件1 <--> 文件2）\n"
    clip_sim_text += "\n".join([
        f"{name1:<20} <--> {name2:<20} : {sim:.4f}"
        for name1, name2, sim in results["clip_similarity_results"]
    ])
    csv_file = results["csv_path"]

    return "",reference_html_str,html_str, image_scores_text, face_sim_text, avg_face_sim_text, clip_sim_text, csv_file


if __name__ == "__main__":
    def score_only_interface(image_folder):
        """
        仅评分（美学+清晰度），不计算人脸和CLIP相似度。
        Args:
            image_folder (list): 上传的图片文件路径列表。
        Returns:
            tuple: 评分展示内容。
        """
        if not image_folder:
            return "未选择图像文件夹。", None
        # 清空 PROJECT_IMAGE_DIR
        if os.path.exists(PROJECT_IMAGE_DIR):
            shutil.rmtree(PROJECT_IMAGE_DIR)
        os.makedirs(PROJECT_IMAGE_DIR, exist_ok=True)
        for src_path in image_folder:
            base_name = os.path.basename(src_path)
            if base_name.endswith(".DS_Store") or 'DS_Store' in base_name:
                continue
            dst_path = os.path.join(PROJECT_IMAGE_DIR, base_name)
            shutil.copy2(src_path, dst_path)

        image_folder_path = PROJECT_IMAGE_DIR
        results = start(image_folder_path, reference_image_name=None)
        if not results:
            return "未找到有效的图像文件。", None

        popup_html = """
        <style>
          .popup-img:hover { cursor: zoom-in; }
          .popup-overlay {
              display: none;
              position: fixed;
              top: 0; left: 0; width: 100%; height: 100%;
              background: rgba(0,0,0,0.7);
              justify-content: center; align-items: center;
              z-index: 9999;
          }
          .popup-overlay img {
              max-width: 90vw; max-height: 90vh;
              box-shadow: 0 0 10px #fff;
          }
        </style>
        <script>
          function enlargeImage(src) {
              const overlay = document.getElementById('popupOverlay');
              const img = document.getElementById('popupImg');
              img.src = src;
              overlay.style.display = 'flex';
          }
          function closeOverlay() {
              document.getElementById('popupOverlay').style.display = 'none';
          }
        </script>
        <div id="popupOverlay" class="popup-overlay" onclick="closeOverlay()">
          <img id="popupImg" src="" />
        </div>
        """

        html_str = "<h3>评分图像展示</h3>" + popup_html
        for name, aes, blur, _ in results["image_scores"]:
            img_path = os.path.join(image_folder_path, name)
            img_data = image_to_base64(img_path)
            html_str += f"""
            <div style='display:inline-block; margin:10px; position:relative;'>
                <img src="{img_data}" style='width:200px; display:block;' class="popup-img" onclick="enlargeImage(this.src)" />
                <div style='position:absolute; top:4px; left:6px; background-color:rgba(0,0,0,0.5); color:white; padding:2px 4px; font-size:12px;'>美学 {aes:.2f}</div>
                <div style='position:absolute; top:24px; left:6px; background-color:rgba(0,0,0,0.5); color:white; padding:2px 4px; font-size:12px;'>清晰 {blur:.2f}</div>
            </div>
            """

        image_scores_text = "\n".join([f"{name}: 美学分 {aes:.2f}, 清晰度 {blur:.2f}" for name, aes, blur, _ in results["image_scores"]])
        return html_str, image_scores_text

    with gr.Blocks() as demo:
        gr.Markdown("### 图像评分系统")
        with gr.Row():
            # image_folder = gr.File(label="选择图像文件夹", file_count="directory")
            # reference_image = gr.File(label="选择参考图像（可选）", file_count="single")
            image_folder = gr.File(
                label="选择图像文件夹",
                file_count="directory",
                # file_types=[".png", ".jpg", ".jpeg", ".webp"],
                type="filepath",
                interactive=True,
                show_label=True
            )
            reference_image = gr.File(
                label="选择参考图像（可选）",
                file_count="single",
                file_types=[".png", ".jpg", ".jpeg", ".webp"],
                type="filepath",
                interactive=True,
                show_label=True
            )
            # image_gallery = gr.Gallery(label="已选图像预览", columns=4, show_label=True, show_download_button=False, allow_preview=True, object_fit="contain")
            image_gallery = gr.Gallery(label="已选图像预览", columns=4, allow_preview=True, show_label=False, elem_id="scored-gallery")
            reference_gallery = gr.Image(label="参考图像预览")

        score_only_btn = gr.Button("仅评分（美学+清晰度）")
        submit_btn = gr.Button("开始评分")
        with gr.Column():
            error_output = gr.HTML(label="错误提示")
            reference_html_output = gr.HTML(label="参考图像评分")
            image_gallery_output = gr.HTML(label="评分图像展示")
            image_scores_output = gr.Textbox(label="单张图像评分结果", lines=10)
            face_sim_output = gr.Textbox(label="全局人脸相似度结果", lines=10)
            avg_face_sim_output = gr.Textbox(label="平均人脸相似度", lines=1)
            clip_sim_output = gr.Textbox(label="CLIP 相似度结果", lines=10)
            csv_output = gr.File(label="下载评分结果 CSV 文件")
        # 图像文件夹和参考图像的联动预览
        # image_folder.change(lambda files: files, inputs=image_folder, outputs=image_gallery)
        def filter_image_files(files):
            valid_exts = (".png", ".jpg", ".jpeg", ".webp")
            ignored_suffixes = (".DS_Store", ".git", ".ignore")
            return [
                [f, os.path.basename(f)]
                for f in files
                if f.lower().endswith(valid_exts) and not f.endswith(ignored_suffixes)
            ]
        image_folder.change(fn=filter_image_files, inputs=image_folder, outputs=image_gallery)
        reference_image.change(lambda img: img, inputs=reference_image, outputs=reference_gallery)

        score_only_btn.click(
            fn=score_only_interface,
            inputs=[image_folder],
            outputs=[image_gallery_output, image_scores_output]
        )

        submit_btn.click(
            fn=gradio_interface,
            inputs=[image_folder, reference_image],
            outputs = [
                error_output,  # 0. error_output
                reference_html_output, # 1. reference_html_str
                image_gallery_output,  # 2. html_str
                image_scores_output,  # 3. image_scores_text
                face_sim_output,  # 4. face_sim_text
                avg_face_sim_output,  # 5. avg_face_sim_text
                clip_sim_output,  # 6. clip_sim_text
                csv_output  # 7. csv_file
            ]
        )


    demo.launch(server_name="0.0.0.0", server_port=7860,)


