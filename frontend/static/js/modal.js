function openModal(imagePath, tileId, tag, score) {
  document.getElementById('modalTileId').innerText = 'IDENTIFIER: ' + tileId;
  document.getElementById('modalImage').src = '/' + imagePath;
  document.getElementById('modalTag').innerText = tag || 'Satellite Patch';
  document.getElementById('modalScore').innerText = score || '0.00';
  document.getElementById('imageModal').classList.remove('hidden');
}

function closeModal() {
  document.getElementById('imageModal').classList.add('hidden');
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});
