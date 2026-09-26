import React from "react";

export function BrandLogo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="MiseEnManagement Logo"
    >
      <path
        d="M6 25L16 6L26 25H20L16 16L12 25H6Z"
        fill="white"
      />
      <path
        d="M16 16L20 25H26L16 6V16Z"
        fill="white"
        fillOpacity="0.85"
      />
    </svg>
  );
}

export function VisaWordmark({ className = "h-4" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 100 32"
      fill="white"
      className={`h-4 sm:h-5 w-auto select-none ${className}`}
      aria-label="VISA"
    >
      <path d="M38.8 3.5l-6.3 18.2h-4.9l3.8-18.2h7.4zm23.9 11.9c0-4.8-6.6-5.1-6.6-7.2 0-2.2 2.5-2.3 4.9-2.3 2.9 0 5.1.6 6.8 1.4l1.2-5.7C66.8.6 63.8 0 60 0c-7.4 0-12.6 3.9-12.6 9.6 0 7.4 10.3 7.8 10.3 11.9 0 2.4-2.9 2.6-5.7 2.6-3.8 0-6.7-.9-8.7-1.8l-1.3 6c2.4 1.1 6.1 1.7 10 1.7 7.7 0 13-3.8 13-9.5zm19.6-11.9l-5.8 18.2h-4.8l5.8-18.2h4.8zm-54.8 0l-7.3 12.4-3-10.8c-.5-1.9-2-2.1-3.6-2.1H.5L0 4.1c3.1.7 6.6 2.5 8.7 5.3l7.6 18.8h5.2l7.9-24.7h-5.9z" />
    </svg>
  );
}

export function AdobeIcon({ className = "w-10 h-10" }: { className?: string }) {
  return (
    <div className={`rounded-xl bg-[#fa0f00] flex items-center justify-center p-2 shadow-sm ${className}`}>
      <svg viewBox="0 0 24 24" fill="white" className="w-full h-full">
        <path d="M13.983 3.6h9.117v16.8l-9.117-16.8zm-3.966 0l-9.117 16.8v-16.8h9.117zm-1.077 11.233l2.06 4.767h3.197l-5.257-12-3.8 8.767h3.8z" />
      </svg>
    </div>
  );
}
