document.getElementById('submitButton').addEventListener('click', async () => {
    const audioFileInput = document.getElementById('audioFile');
    const file = audioFileInput.files[0];
    
    if (file) {
        const formData = new FormData();
        formData.append('audio_file', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.text();
        document.getElementById('result').value = result;
    } else {
        alert('Please select an audio file.');
    }
});

document.getElementById('clearButton').addEventListener('click', () => {
    document.getElementById('audioFile').value = '';
    document.getElementById('result').value = '';
});
