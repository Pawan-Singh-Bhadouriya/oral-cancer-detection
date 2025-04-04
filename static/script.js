document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileUpload");
    const status = document.getElementById("uploadStatus");
    const preview = document.getElementById("previewImage");
  
    uploadForm.addEventListener("submit", function (event) {
      event.preventDefault();
  
      const file = fileInput.files[0];
      if (!file) {
        status.innerText = "Please select a file!";
        status.style.color = "red";
        return;
      }
  
      const reader = new FileReader();
      reader.onload = function (e) {
        preview.src = e.target.result;
        preview.style.display = "block";
      };
      reader.readAsDataURL(file);
  
      const formData = new FormData();
      formData.append("file", file);
      fetch("/upload", {
        method: "POST",
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => {
          if (data.status === "success") {
            const isCancer = data.prediction === "Cancer";
            const predictionColor = isCancer ? "red" : "green";
            status.innerHTML = `Image uploaded successfully!<br><strong style="color: ${predictionColor};">Result: ${data.prediction}</strong>`;
          } else {
            status.innerText = data.message;
            status.style.color = "red";
          }
        })
        .catch(() => {
          status.innerText = "Error uploading file!";
          status.style.color = "red";
        });
    });
  });
  