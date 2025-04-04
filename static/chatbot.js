document.getElementById('chatForm').addEventListener('submit', function(e) {
    e.preventDefault();
  
    const input = document.getElementById('userInput');
    const message = input.value;
    const chatBox = document.getElementById('chatBox');
  
    // User message
    chatBox.innerHTML += `<div><strong>You:</strong> ${message}</div>`;
  
    // Fake bot response
    setTimeout(() => {
      fetch('/chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })
        .then((res) => res.json())
        .then((data) => {
          chatBox.innerHTML += `<div><strong>Bot:</strong> ${data.response}</div>`;
          chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(() => {
          chatBox.innerHTML += `<div><strong>Bot:</strong> Error fetching response.</div>`;
        });
    }, 500);
  
    input.value = '';
  });
  