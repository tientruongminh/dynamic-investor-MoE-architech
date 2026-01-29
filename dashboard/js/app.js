// MoE Alpha Dashboard - JavaScript

// Page Navigation
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });

    // Remove active from all nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });

    // Show selected page
    const page = document.getElementById(pageId);
    if (page) {
        page.classList.add('active');
    }

    // Activate nav item
    const navItem = document.querySelector(`a[href="#${pageId}"]`);
    if (navItem) {
        navItem.parentElement.classList.add('active');
    }

    // Update URL hash
    window.location.hash = pageId;
}

// Handle URL hash on load
window.addEventListener('load', () => {
    const hash = window.location.hash.slice(1) || 'overview';
    showPage(hash);
});

// Handle browser back/forward
window.addEventListener('hashchange', () => {
    const hash = window.location.hash.slice(1) || 'overview';
    showPage(hash);
});

// Animate metrics on scroll
const observerOptions = {
    threshold: 0.1
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate');
        }
    });
}, observerOptions);

document.querySelectorAll('.metric-card').forEach(card => {
    observer.observe(card);
});

// Console welcome message
console.log('%c🚀 MoE Alpha Framework Dashboard', 'color: #4f46e5; font-size: 20px; font-weight: bold;');
console.log('%cVersion 1.0 - 2026', 'color: #64748b; font-size: 12px;');
