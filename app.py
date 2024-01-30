from flask import Flask, render_template, request, send_file
import os

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = '/output/images'

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "photo" in request.files:
        photo = request.files["photo"]
        if photo:
            # Set the destination directory
            upload_folder = os.path.join(app.root_path, "static")
            os.makedirs(upload_folder, exist_ok=True)  # Ensure the folder exists

            # Save the uploaded photo
            filename = os.path.join(upload_folder, photo.filename)
            photo.save(filename)

            # Print the filename on the console
            print("Uploaded image filename:", photo.filename)

            return render_template("result.html", input_image=photo.filename, output_image = photo.filename[:-4]+"_modified.png")
            #return render_template("result.html", input_image="original.jpg", output_image="modified.jpg",persons_detected=persons_count,paintings_detected= paintings_count)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')