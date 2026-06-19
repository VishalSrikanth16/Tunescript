function BrandMark({ className = '' }) {
  const classes = ['brand-mark', className].filter(Boolean).join(' ');

  return (
    <div className={classes} aria-hidden="true">
      <span />
      <span />
      <span />
      <span />
      <span />
    </div>
  );
}

export default BrandMark;
