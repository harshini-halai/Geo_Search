function setQuery(text) {
  document.getElementById('searchInput').value = text;
  executeSearch();
}

async function executeSearch() {
  const q = document.getElementById('searchInput').value;
  if (!q.trim()) return;
  
  const container = document.getElementById('searchResults');
  container.innerHTML = `
    <div class="col-span-3 py-6 rounded-xl bg-slate-900/40 border border-slate-800 text-center font-mono text-xs text-sky-400 animate-pulse">
      Computing cross-attention cosine distance via OpenCLIP...
    </div>
  `;

  try {
    const res = await fetch(`/api/v1/search/text?query=${encodeURIComponent(q)}&limit=3`);
    const data = await res.json();
    container.innerHTML = '';
    
    const results = data.results || [];
    if (results.length === 0) {
      container.innerHTML = '<p class="col-span-3 text-slate-500 font-mono text-xs text-center py-4">No matching vectors found.</p>';
      return;
    }

    results.forEach((item, idx) => {
      const scorePercent = item.score ? (item.score * 100).toFixed(1) : 'N/A';
      container.innerHTML += `
        <div class="bg-[#080d19] p-3 rounded-xl border border-slate-800 flex flex-col justify-between hover:border-sky-500/50 transition shadow-lg">
          <div class="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-2">
            <span>MATCH #${idx + 1}</span>
            <span class="text-emerald-400 font-bold">${scorePercent}% Match</span>
          </div>
          <p class="font-mono text-[11px] text-slate-200 truncate mb-2">${item.image_path || item.id}</p>
          <div class="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
            <div class="bg-gradient-to-r from-sky-400 to-teal-400 h-full" style="width: ${scorePercent}%"></div>
          </div>
        </div>
      `;
    });
  } catch (err) {
    container.innerHTML = '<p class="col-span-3 text-rose-400 font-mono text-xs text-center py-4">Error executing vector query.</p>';
  }
}

document.addEventListener("DOMContentLoaded", () => {
    // Search input field aur rows target karo
    const searchInput = document.querySelector('input[type="search"]') 
                     || document.querySelector('#searchInput')
                     || document.querySelector('input[placeholder*="Search"]');

    if (!searchInput) return;

    searchInput.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().trim();
        const rows = document.querySelectorAll("table tbody tr");

        rows.forEach((row) => {
            const rowText = row.innerText.toLowerCase();
            // Agar query match kare toh dikhao, warna hide karo
            if (rowText.includes(query)) {
                row.style.display = "";
            } else {
                row.style.display = "none";
            }
        });
    });
});
