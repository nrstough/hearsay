import React from "react";

interface DottedNumberProps {
  value: string | number;
  className?: string;
  subduedDecimals?: boolean;
  subduedClassName?: string;
  currencyClassName?: string;
}

/**
 * Renders digits in the Doto dot-matrix font while rendering
 * punctuation, currency symbols, and mathematical signs in Author typography.
 */
export function DottedNumber({
  value,
  className = "",
  subduedDecimals = false,
  subduedClassName = "text-slate-400 font-normal",
  currencyClassName = "",
}: DottedNumberProps) {
  const rawStr = typeof value === "number" ? value.toString() : value;
  const str = rawStr.replace(/,/g, "");

  if (subduedDecimals && str.includes(".")) {
    const dotIndex = str.indexOf(".");
    const wholePart = str.slice(0, dotIndex);
    const decimalDigits = str.slice(dotIndex + 1);

    return (
      <span className={`inline-flex items-baseline select-none ${className}`}>
        <FormattedPart str={wholePart} currencyClassName={currencyClassName} />
        <span className={`inline-flex items-baseline ${subduedClassName}`}>
          <span
            className="font-sans font-bold text-[0.75em] mx-[1.5px] leading-none select-none"
            style={{ verticalAlign: "baseline" }}
          >
            .
          </span>
          <FormattedPart str={decimalDigits} />
        </span>
      </span>
    );
  }

  return (
    <span className={`inline-flex items-baseline select-none ${className}`}>
      <FormattedPart str={str} currencyClassName={currencyClassName} />
    </span>
  );
}

function FormattedPart({
  str,
  currencyClassName = "",
}: {
  str: string;
  currencyClassName?: string;
}) {
  const tokens = str.split(/([0-9]+)/).filter(Boolean);

  return (
    <>
      {tokens.map((token, idx) => {
        const isDigits = /^[0-9]+$/.test(token);

        if (isDigits) {
          return (
            <span key={idx} className="font-dotted font-bold tracking-tight">
              {token}
            </span>
          );
        }

        return (
          <span key={idx} className="inline-flex items-baseline">
            {token.split("").map((char, charIdx) => {
              if (char === ",") {
                return (
                  <span
                    key={charIdx}
                    className="font-sans font-semibold text-[0.8em] -mx-[0.5px] leading-none select-none"
                    style={{ verticalAlign: "baseline" }}
                  >
                    ,
                  </span>
                );
              }

              if (char === ".") {
                return (
                  <span
                    key={charIdx}
                    className="font-sans font-bold text-[0.75em] mx-[1.5px] leading-none select-none"
                    style={{ verticalAlign: "baseline" }}
                  >
                    .
                  </span>
                );
              }

              if (char === "$") {
                return (
                  <span
                    key={charIdx}
                    className={`font-sans font-medium text-[0.78em] leading-none inline-block relative -top-[0.05em] mr-[3px] select-none ${currencyClassName}`}
                  >
                    $
                  </span>
                );
              }

              if (char === "-") {
                return (
                  <span
                    key={charIdx}
                    className="font-sans font-semibold text-[0.85em] mr-[2px] leading-none inline-block relative -top-[0.03em] select-none"
                  >
                    -
                  </span>
                );
              }

              if (char === "+") {
                return (
                  <span
                    key={charIdx}
                    className="font-sans font-semibold text-[0.85em] mr-[2px] leading-none inline-block relative -top-[0.03em] select-none"
                  >
                    +
                  </span>
                );
              }

              if (char === "%") {
                return (
                  <span
                    key={charIdx}
                    className="font-sans font-semibold text-[0.76em] ml-[1.5px] leading-none inline-block relative -top-[0.02em] select-none"
                  >
                    %
                  </span>
                );
              }

              return (
                <span key={charIdx} className="font-sans font-normal leading-none select-none">
                  {char}
                </span>
              );
            })}
          </span>
        );
      })}
    </>
  );
}
